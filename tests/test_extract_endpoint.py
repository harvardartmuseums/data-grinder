import json
import pytest
from unittest.mock import patch, MagicMock

MOCK_RESULT = {"status": "ok", "url": "https://example.com/iiif/item/info.json", "width": 100, "height": 100}
VALID_URL = "https://images.example.com/iiif/item123"


# ── Smoke tests ───────────────────────────────────────────────────────────────

def test_home(client):
    r = client.get("/")
    assert r.status_code == 200
    assert r.get_json()["status"] == "ok"

def test_list_services_structure(client):
    r = client.get("/list/services")
    assert r.status_code == 200
    data = r.get_json()
    assert "services" in data
    services = data["services"]
    assert "computer vision" in services
    assert "large language models" in services
    assert "other" in services


# ── Missing parameter handling ────────────────────────────────────────────────

def test_no_params_returns_missing_message(client):
    r = client.get("/extract")
    assert r.status_code == 200
    assert "missing" in r.get_json()["status"]

def test_url_only_returns_missing_message(client):
    r = client.get(f"/extract?url={VALID_URL}")
    assert r.status_code == 200
    assert "missing" in r.get_json()["status"]

def test_services_only_returns_missing_message(client):
    r = client.get("/extract?services=imagga")
    assert r.status_code == 200
    assert "missing" in r.get_json()["status"]


# ── Successful routing ────────────────────────────────────────────────────────

def test_valid_request_calls_process_image(client):
    with patch("main.process_image", return_value=MOCK_RESULT) as mock_fn:
        r = client.get(f"/extract?url={VALID_URL}&services=imagga")
        assert r.status_code == 200
        assert r.get_json() == MOCK_RESULT
        mock_fn.assert_called_once()

def test_services_parsed_before_process_image(client):
    with patch("main.process_image", return_value=MOCK_RESULT) as mock_fn:
        client.get(f"/extract?url={VALID_URL}&services=imagga:tags|faces")
        _, call_kwargs = mock_fn.call_args
        # process_image(url, services, prompt=...)
        call_args = mock_fn.call_args[0]
        assert call_args[1] == {"imagga": ["tags", "faces"]}

def test_prompt_passed_through(client):
    with patch("main.process_image", return_value=MOCK_RESULT) as mock_fn:
        client.get(f"/extract?url={VALID_URL}&services=imagga&prompt=Describe")
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs.get("prompt") == "Describe"


# ── Prompt validation ─────────────────────────────────────────────────────────

def test_prompt_too_long_returns_400(client):
    long_prompt = "a" * 501
    r = client.get(f"/extract?url={VALID_URL}&services=imagga&prompt={long_prompt}")
    assert r.status_code == 400
    assert "500" in r.get_json()["status"]

def test_prompt_max_length_accepted(client):
    max_prompt = "a" * 500
    with patch("main.process_image", return_value=MOCK_RESULT):
        r = client.get(f"/extract?url={VALID_URL}&services=imagga&prompt={max_prompt}")
        assert r.status_code == 200

def test_control_chars_stripped_before_process(client):
    with patch("main.process_image", return_value=MOCK_RESULT) as mock_fn:
        client.get(f"/extract?url={VALID_URL}&services=imagga&prompt=hello%00world")
        call_kwargs = mock_fn.call_args[1]
        assert call_kwargs.get("prompt") == "helloworld"
