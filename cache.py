import os
import re
import time
import logging
import datetime

logger = logging.getLogger(__name__)
import requests
from urllib.parse import urlparse
from PIL import Image
from dotenv import load_dotenv

load_dotenv()

CACHE_DAYS  = int(os.getenv("IMAGE_CACHE_DAYS", "30"))
_local_root = os.getenv("IMAGE_CACHE_DIR") or os.path.dirname(os.path.realpath(__file__)) + "/temp"
_s3_bucket  = os.getenv("IMAGE_CACHE_S3_BUCKET", "")
_s3_prefix  = os.getenv("IMAGE_CACHE_S3_PREFIX", "")

USER_AGENT = os.getenv("USER_AGENT", "data-grinder/1.0")

_s3 = None


def _parse_sizes(raw):
    sizes = []
    for s in raw.split(","):
        s = s.strip()
        if s:
            sizes.append(s)
    if "full" not in sizes:
        sizes.insert(0, "full")
    return sizes


# e.g. ["full", "1110", "512"] — always includes "full"
CACHE_SIZES = _parse_sizes(os.getenv("IMAGE_CACHE_SIZES", "full"))


def _s3_client():
    global _s3
    if _s3 is None:
        import boto3
        _s3 = boto3.client(
            "s3",
            region_name=os.getenv("AWS_REGION"),
            aws_access_key_id=os.getenv("AWS_ACCESS_KEY"),
            aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
        )
    return _s3


def cache_names_from_url(input_url):
    """Return (domain, basename) derived from the IIIF input URL."""
    parsed = urlparse(input_url)
    domain = parsed.netloc
    last_segment = [s for s in parsed.path.split("/") if s][-1] if parsed.path else "image"
    basename = re.sub(r"[^a-zA-Z0-9._-]", "_", last_segment)
    return domain, basename


def _local_path(tier, domain, basename):
    folder = os.path.join(_local_root, tier, domain)
    os.makedirs(folder, exist_ok=True)
    return os.path.join(folder, basename + ".jpg")


def _s3_key(tier, domain, basename):
    return f"{_s3_prefix}{tier}/{domain}/{basename}.jpg"


def _s3_age_seconds(client, key):
    from botocore.exceptions import ClientError
    try:
        head = client.head_object(Bucket=_s3_bucket, Key=key)
        age = (datetime.datetime.now(datetime.timezone.utc) - head["LastModified"]).total_seconds()
        return age
    except ClientError as e:
        if e.response["Error"]["Code"] in ("404", "NoSuchKey"):
            return float("inf")
        logger.warning("S3 head_object failed for %s: %s", key, e)
        return float("inf")


def _s3_download(client, key, local_path):
    from botocore.exceptions import ClientError
    try:
        response = client.get_object(Bucket=_s3_bucket, Key=key)
        with open(local_path, "wb") as f:
            f.write(response["Body"].read())
        return True
    except ClientError as e:
        logger.warning("S3 get_object failed for %s: %s", key, e)
        return False


def _s3_upload(client, key, local_path):
    from botocore.exceptions import ClientError
    try:
        with open(local_path, "rb") as f:
            client.put_object(Bucket=_s3_bucket, Key=key, Body=f)
    except ClientError as e:
        logger.warning("S3 put_object failed for %s: %s", key, e)


def _is_fresh(path, cache_days):
    return os.path.exists(path) and (time.time() - os.path.getmtime(path)) < cache_days * 86400


def get_image(download_url, input_url, cache_days):
    """Download and cache all size variants. Returns a dict keyed by tier name.

    On success:
        {
            "status": "ok",
            "full":  {"path": "...", "width": 5000, "height": 4000},
            "1110":  {"path": "...", "width": 1110, "height":  888},
            "512":   {"path": "...", "width":  512, "height":  410},
        }
    On failure:
        {"status": "bad"}
    """
    domain, basename = cache_names_from_url(input_url)
    result = {"status": "bad"}

    # ── Full-size: local → S3 → origin ──────────────────────────────────────
    full_path = _local_path("full", domain, basename)

    if not _is_fresh(full_path, cache_days):
        if _s3_bucket:
            client = _s3_client()
            if _s3_age_seconds(client, _s3_key("full", domain, basename)) < cache_days * 86400:
                _s3_download(client, _s3_key("full", domain, basename), full_path)

        if not _is_fresh(full_path, cache_days):
            try:
                r = requests.get(download_url, headers={"User-Agent": USER_AGENT}, timeout=21)
            except requests.exceptions.RequestException as e:
                logger.warning("origin_download_failed", extra={"url": download_url, "error": str(e)})
                return result
            if r.status_code != 200:
                return result
            with open(full_path, "wb") as f:
                for chunk in r.iter_content(chunk_size=128):
                    f.write(chunk)
            try:
                with Image.open(full_path) as _check:
                    _check.verify()
            except Exception:
                os.remove(full_path)
                return result
            if _s3_bucket:
                _s3_upload(_s3_client(), _s3_key("full", domain, basename), full_path)

    if not os.path.exists(full_path):
        return result

    try:
        full_im = Image.open(full_path)
    except Exception:
        logger.warning("Corrupt cached image, removing: %s", full_path)
        os.remove(full_path)
        return result

    with full_im as im:
        result["status"] = "ok"
        result["full"] = {"path": full_path, "width": im.width, "height": im.height}

        # ── Scaled variants ──────────────────────────────────────────────────────
        for size_str in CACHE_SIZES:
            if size_str == "full":
                continue
            size = int(size_str)
            path = _local_path(size_str, domain, basename)

            if not _is_fresh(path, cache_days):
                restored = False
                if _s3_bucket:
                    client = _s3_client()
                    if _s3_age_seconds(client, _s3_key(size_str, domain, basename)) < cache_days * 86400:
                        restored = _s3_download(client, _s3_key(size_str, domain, basename), path)
                if not restored:
                    scaled = im.copy()
                    scaled.thumbnail((size, size), Image.LANCZOS)
                    scaled.save(path)
                    scaled.close()
                    if _s3_bucket:
                        _s3_upload(_s3_client(), _s3_key(size_str, domain, basename), path)

            try:
                with Image.open(path) as scaled_im:
                    result[size_str] = {"path": path, "width": scaled_im.width, "height": scaled_im.height}
            except Exception:
                logger.warning("Corrupt cached scaled image, removing: %s", path)
                os.remove(path)

    return result
