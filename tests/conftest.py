import io
import os
import pytest
from unittest.mock import patch
from PIL import Image


@pytest.fixture(scope="session")
def app():
    env = {
        "IMAGE_CACHE_DIR": "/tmp/dg_test_cache",
        "IMAGE_CACHE_S3_BUCKET": "",
        "IMAGE_CACHE_SIZES": "full",
        "AWS_ACCESS_KEY": "AKIATEST",
        "AWS_SECRET_ACCESS_KEY": "secret",
        "AWS_REGION": "us-east-1",
        "IMAGGA_KEY": "k",
        "IMAGGA_SECRET": "s",
        "CLARIFAI_PAT": "p",
        "GOOGLE_API_KEY": "g",
    }
    with patch.dict(os.environ, env):
        import main as m
        m.app.config["TESTING"] = True
        yield m.app


@pytest.fixture()
def client(app):
    return app.test_client()


@pytest.fixture(scope="session")
def sample_jpeg_bytes():
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(255, 0, 0)).save(buf, format="JPEG")
    return buf.getvalue()
