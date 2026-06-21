import time
import pytest
from unittest.mock import patch, MagicMock


VALID_URL = "https://images.example.com/iiif/item123"


@pytest.fixture()
def budget_client(app):
    """Client with a very short REQUEST_BUDGET to trigger timeout in tests."""
    import os
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
        "REQUEST_BUDGET": "2",
    }
    with patch.dict(os.environ, env):
        import main
        original = main.REQUEST_BUDGET
        main.REQUEST_BUDGET = 2
        yield app.test_client()
        main.REQUEST_BUDGET = original


def _mock_iiif_image():
    iiif_img = MagicMock()
    iiif_img.status = "ok"
    iiif_img.is_valid.return_value = True
    iiif_img.get_full_image_url.return_value = "http://example.com/full.jpg"
    iiif_img.get_base_uri.return_value = "http://example.com"
    iiif_img.info = {"width": 100, "height": 100}
    return iiif_img


def _mock_cache():
    return {
        "status": "ok",
        "full": {"path": "/tmp/fake.jpg", "width": 100, "height": 100},
        "1110": {"path": "/tmp/fake.jpg", "width": 100, "height": 100},
        "512": {"path": "/tmp/fake.jpg", "width": 100, "height": 100},
    }


def _slow_fetch_model(*args, **kwargs):
    time.sleep(5)
    return "slow_model", {"status": 200, "body": "result"}


def _fast_fetch_model(model, *args, **kwargs):
    return model.name, {"status": 200, "body": "fast result", "runtime": 0.01, "annotationFragment": ""}


class TestRequestBudget:

    def _patch_infra(self):
        """Patch IIIF and cache so process_image reaches the LLM dispatch."""
        return [
            patch("main.iiif.IIIFImage", return_value=_mock_iiif_image()),
            patch("main.cache.get_image", return_value=_mock_cache()),
        ]

    def test_slow_models_do_not_crash_worker(self, budget_client):
        """Models exceeding REQUEST_BUDGET are skipped gracefully."""
        mock_model = MagicMock()
        mock_model.name = "slow-model"

        patches = self._patch_infra() + [
            patch("main.GENERIC_MODELS", [(mock_model, MagicMock, "full")]),
            patch("main._fetch_model", side_effect=_slow_fetch_model),
        ]

        for p in patches:
            p.start()
        try:
            r = budget_client.get(f"/extract?url={VALID_URL}&services=slow-model")
            data = r.get_json()
            assert r.status_code == 200
            assert "slow-model" not in data
        finally:
            for p in patches:
                p.stop()

    def test_fast_models_still_returned(self, budget_client):
        """Models that finish before the budget are included in the response."""
        mock_model = MagicMock()
        mock_model.name = "fast-model"

        patches = self._patch_infra() + [
            patch("main.GENERIC_MODELS", [(mock_model, MagicMock, "full")]),
            patch("main._fetch_model", side_effect=_fast_fetch_model),
        ]

        for p in patches:
            p.start()
        try:
            r = budget_client.get(f"/extract?url={VALID_URL}&services=fast-model")
            data = r.get_json()
            assert r.status_code == 200
            assert "fast-model" in data
            assert data["fast-model"]["body"] == "fast result"
        finally:
            for p in patches:
                p.stop()

    def test_mix_of_fast_and_slow_returns_partial(self, budget_client):
        """When some models finish and others time out, we get partial results."""
        fast_model = MagicMock()
        fast_model.name = "fast-model"
        slow_model = MagicMock()
        slow_model.name = "slow-model"

        def _mixed_fetch(model, *args, **kwargs):
            if model.name == "slow-model":
                time.sleep(5)
                return model.name, {"status": 200, "body": "late"}
            return model.name, {"status": 200, "body": "quick", "runtime": 0.01, "annotationFragment": ""}

        patches = self._patch_infra() + [
            patch("main.GENERIC_MODELS", [(fast_model, MagicMock, "full"), (slow_model, MagicMock, "full")]),
            patch("main._fetch_model", side_effect=_mixed_fetch),
        ]

        for p in patches:
            p.start()
        try:
            r = budget_client.get(f"/extract?url={VALID_URL}&services=fast-model,slow-model")
            data = r.get_json()
            assert r.status_code == 200
            assert "fast-model" in data
            assert "slow-model" not in data
        finally:
            for p in patches:
                p.stop()
