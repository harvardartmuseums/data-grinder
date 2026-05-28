"""
Integration tests that hit live APIs with real credentials.

Skipped by default. To run:

    RUN_INTEGRATION=1 pytest tests/test_integration.py -v

Requires environment variables for the services under test (e.g. GOOGLE_API_KEY).
"""
import os
import tempfile
import pytest
import requests
from dotenv import load_dotenv

load_dotenv()

RUN_INTEGRATION = os.getenv("RUN_INTEGRATION") == "1"
skip_unless_integration = pytest.mark.skipif(
    not RUN_INTEGRATION,
    reason="Set RUN_INTEGRATION=1 to run integration tests"
)


# ── fixtures ──────────────────────────────────────────────────────────────────

def _download_iiif(url):
    image_url = url.rstrip("/") + "/full/full/0/default.jpg"
    resp = requests.get(image_url, timeout=30)
    resp.raise_for_status()
    with tempfile.NamedTemporaryFile(suffix=".jpg", delete=False) as f:
        f.write(resp.content)
        return f.name


@pytest.fixture(scope="module")
def flagged_image_file():
    """Download the known content-flagged image to a temp file once per module."""
    return _download_iiif("https://nrs.harvard.edu/urn-3:HUAM:783070")


@pytest.fixture(scope="module")
def safe_image_file():
    """Download a known safe image to a temp file once per module."""
    return _download_iiif("https://nrs.harvard.edu/urn-3:HUAM:797665")


def _assert_clean_response(result, model_label):
    """Assert a normal successful response with no filtering flags set."""
    assert result["status"] == 200, f"{model_label}: expected status 200, got {result['status']}"
    assert result["body"] is not None, f"{model_label}: expected non-null body"
    assert not result.get("filtered"), f"{model_label}: unexpected 'filtered' flag"
    assert not result.get("content_policy_violation"), f"{model_label}: unexpected 'content_policy_violation' flag"


def _assert_content_blocked(result, model_label):
    """Assert that exactly one content-blocking flag is set and body is null."""
    filtered = result.get("filtered", False)
    policy_violation = result.get("content_policy_violation", False)
    assert filtered or policy_violation, (
        f"{model_label}: expected 'filtered' or 'content_policy_violation' to be True, "
        f"got: {result}"
    )
    assert result["body"] is None, (
        f"{model_label}: expected body to be null when content is blocked, got: {result['body']!r}"
    )
    flag = "filtered" if filtered else "content_policy_violation"
    print(f"\n{model_label}: blocked via '{flag}'")


# ── Google Gemini ─────────────────────────────────────────────────────────────

@skip_unless_integration
@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
class TestGeminiContentBlock:

    def test_flash_2_0(self, flagged_image_file):
        from parsers.googlegemini import GoogleGemini, GoogleGeminiModel
        result = GoogleGemini().fetch(flagged_image_file, model=GoogleGeminiModel.FLASH_2_0)
        _assert_content_blocked(result, "gemini-2-0-flash")

    def test_flash_lite_2_0(self, flagged_image_file):
        from parsers.googlegemini import GoogleGemini, GoogleGeminiModel
        result = GoogleGemini().fetch(flagged_image_file, model=GoogleGeminiModel.FLASH_LITE_2_0)
        _assert_content_blocked(result, "gemini-2-0-flash-lite")

    def test_flash_2_5(self, flagged_image_file):
        from parsers.googlegemini import GoogleGemini, GoogleGeminiModel
        result = GoogleGemini().fetch(flagged_image_file, model=GoogleGeminiModel.FLASH_2_5)
        _assert_content_blocked(result, "gemini-2-5-flash")

    def test_flash_lite_2_5(self, flagged_image_file):
        from parsers.googlegemini import GoogleGemini, GoogleGeminiModel
        result = GoogleGemini().fetch(flagged_image_file, model=GoogleGeminiModel.FLASH_LITE_2_5)
        _assert_content_blocked(result, "gemini-2-5-flash-lite")


@skip_unless_integration
@pytest.mark.skipif(not os.getenv("GOOGLE_API_KEY"), reason="GOOGLE_API_KEY not set")
class TestGeminiSafeImage:

    def test_flash_2_0(self, safe_image_file):
        from parsers.googlegemini import GoogleGemini, GoogleGeminiModel
        result = GoogleGemini().fetch(safe_image_file, model=GoogleGeminiModel.FLASH_2_0)
        _assert_clean_response(result, "gemini-2-0-flash")

    def test_flash_lite_2_0(self, safe_image_file):
        from parsers.googlegemini import GoogleGemini, GoogleGeminiModel
        result = GoogleGemini().fetch(safe_image_file, model=GoogleGeminiModel.FLASH_LITE_2_0)
        _assert_clean_response(result, "gemini-2-0-flash-lite")

    def test_flash_2_5(self, safe_image_file):
        from parsers.googlegemini import GoogleGemini, GoogleGeminiModel
        result = GoogleGemini().fetch(safe_image_file, model=GoogleGeminiModel.FLASH_2_5)
        _assert_clean_response(result, "gemini-2-5-flash")

    def test_flash_lite_2_5(self, safe_image_file):
        from parsers.googlegemini import GoogleGemini, GoogleGeminiModel
        result = GoogleGemini().fetch(safe_image_file, model=GoogleGeminiModel.FLASH_LITE_2_5)
        _assert_clean_response(result, "gemini-2-5-flash-lite")
