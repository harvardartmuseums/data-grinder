# NOTE: cache.py reads _local_root, _s3_bucket, and CACHE_SIZES at import time
# from os.getenv. Tests that need different values must call importlib.reload(cache)
# AFTER patching env vars with monkeypatch.setenv.

import importlib
import io
import os
import datetime

import pytest
from PIL import Image


DOWNLOAD_URL = "https://images.example.com/full/full/0/default.jpg"
INPUT_URL = "https://images.example.com/iiif/item123"


def _make_jpeg(size=(10, 10)):
    buf = io.BytesIO()
    Image.new("RGB", size, color=(255, 0, 0)).save(buf, format="JPEG")
    return buf.getvalue()


# ── cache_names_from_url ──────────────────────────────────────────────────────

from cache import cache_names_from_url

def test_cache_names_standard_url():
    domain, basename = cache_names_from_url("https://nrs.harvard.edu/urn-3:HUAM:DDC250728")
    assert domain == "nrs.harvard.edu"
    assert basename == "urn-3_HUAM_DDC250728"

def test_cache_names_trailing_slash():
    domain, basename = cache_names_from_url("https://example.com/images/test/")
    assert domain == "example.com"
    assert basename == "test"

def test_cache_names_percent_encoded():
    domain, basename = cache_names_from_url("https://example.com/images/test%20file.jpg")
    assert domain == "example.com"
    assert basename == "test_20file.jpg"

def test_cache_names_simple_path():
    domain, basename = cache_names_from_url("https://example.com/abc")
    assert domain == "example.com"
    assert basename == "abc"


# ── get_image ─────────────────────────────────────────────────────────────────

def _reload_cache(monkeypatch, tmp_path, sizes="full", s3_bucket=""):
    monkeypatch.setenv("IMAGE_CACHE_DIR", str(tmp_path))
    monkeypatch.setenv("IMAGE_CACHE_S3_BUCKET", s3_bucket)
    monkeypatch.setenv("IMAGE_CACHE_SIZES", sizes)
    import cache
    importlib.reload(cache)
    return cache


def test_fresh_download(tmp_path, monkeypatch, requests_mock):
    cache = _reload_cache(monkeypatch, tmp_path)
    requests_mock.get(DOWNLOAD_URL, content=_make_jpeg())

    result = cache.get_image(DOWNLOAD_URL, INPUT_URL, cache_days=30)

    assert result["status"] == "ok"
    assert os.path.exists(result["full"]["path"])
    assert result["full"]["width"] == 10
    assert result["full"]["height"] == 10


def test_fresh_local_cache_not_re_downloaded(tmp_path, monkeypatch, requests_mock):
    cache = _reload_cache(monkeypatch, tmp_path)
    requests_mock.get(DOWNLOAD_URL, content=_make_jpeg())

    # First call downloads
    cache.get_image(DOWNLOAD_URL, INPUT_URL, cache_days=30)
    call_count_after_first = requests_mock.call_count

    # Second call should serve from local cache
    result = cache.get_image(DOWNLOAD_URL, INPUT_URL, cache_days=30)
    assert result["status"] == "ok"
    assert requests_mock.call_count == call_count_after_first


def test_origin_404_returns_bad(tmp_path, monkeypatch, requests_mock):
    cache = _reload_cache(monkeypatch, tmp_path)
    requests_mock.get(DOWNLOAD_URL, status_code=404)

    result = cache.get_image(DOWNLOAD_URL, INPUT_URL, cache_days=30)
    assert result == {"status": "bad"}


def test_corrupt_image_returns_bad(tmp_path, monkeypatch, requests_mock):
    cache = _reload_cache(monkeypatch, tmp_path)
    requests_mock.get(DOWNLOAD_URL, content=b"not an image at all")

    result = cache.get_image(DOWNLOAD_URL, INPUT_URL, cache_days=30)
    assert result == {"status": "bad"}


def test_scaled_variants_generated(tmp_path, monkeypatch, requests_mock):
    cache = _reload_cache(monkeypatch, tmp_path, sizes="full,1110")
    # Use a larger image so thumbnail actually produces a 1110px-wide result
    requests_mock.get(DOWNLOAD_URL, content=_make_jpeg(size=(2000, 1500)))

    result = cache.get_image(DOWNLOAD_URL, INPUT_URL, cache_days=30)

    assert result["status"] == "ok"
    assert "1110" in result
    assert result["1110"]["width"] <= 1110
    assert result["1110"]["height"] <= 1110
    assert os.path.exists(result["1110"]["path"])


def test_s3_fresh_skips_origin(tmp_path, monkeypatch, requests_mock):
    from unittest.mock import MagicMock, patch
    import io as _io

    cache = _reload_cache(monkeypatch, tmp_path, s3_bucket="test-bucket")

    jpeg_bytes = _make_jpeg()
    recent_date = datetime.datetime.now(datetime.timezone.utc) - datetime.timedelta(seconds=60)

    mock_s3 = MagicMock()
    mock_s3.head_object.return_value = {"LastModified": recent_date}
    mock_s3.get_object.return_value = {"Body": _io.BytesIO(jpeg_bytes)}

    with patch("cache._s3_client", return_value=mock_s3):
        # _s3 global must also be reset so _s3_client() is called
        cache._s3 = None
        result = cache.get_image(DOWNLOAD_URL, INPUT_URL, cache_days=30)

    assert result["status"] == "ok"
    assert requests_mock.call_count == 0
