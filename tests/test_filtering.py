"""Tests for content-filter and policy-violation flag handling in Bedrock and Gemini parsers."""
import io
import os
import pytest
from unittest.mock import MagicMock, patch
from PIL import Image


@pytest.fixture()
def sample_jpeg(tmp_path):
    buf = io.BytesIO()
    Image.new("RGB", (10, 10), color=(128, 128, 128)).save(buf, format="JPEG")
    p = tmp_path / "sample.jpg"
    p.write_bytes(buf.getvalue())
    return str(p)


# ── helpers ───────────────────────────────────────────────────────────────────

def _make_bedrock_response(stop_reason, text="some text"):
    resp = {"stopReason": stop_reason}
    if stop_reason != "content_filtered":
        resp["output"] = {"message": {"content": [{"text": text}]}}
    return resp


# ── AWS Bedrock parsers ───────────────────────────────────────────────────────

BEDROCK_PARSERS = [
    ("parsers.awsanthropic", "AWSAnthropic", "AnthropicModel", "CLAUDE_3_HAIKU"),
    ("parsers.awsmeta",      "AWSMeta",      "MetaModel",      "LLAMA_3_2_11B"),
    ("parsers.awsnova",      "AWSNova",      "NovaModel",      "NOVA_LITE_1_0"),
    ("parsers.awsmistral",   "AWSMistral",   "MistralModel",   "PIXTRAL_LARGE_2502"),
    ("parsers.awsqwen",      "AWSQwen",      "QwenModel",      "QWEN_3_VL_235B"),
    ("parsers.awsmoonshot",  "AWSMoonshot",  "MoonshotModel",  "KIMI_K_2_5"),
    ("parsers.awswriter",    "AWSWriter",    "WriterModel",    "PALMYRA_VISION_7B"),
]


@pytest.mark.parametrize("module,cls,model_cls,model_member", BEDROCK_PARSERS)
def test_bedrock_content_filtered(module, cls, model_cls, model_member, sample_jpeg):
    import importlib
    mod = importlib.import_module(module)
    parser_cls = getattr(mod, cls)
    model_enum = getattr(mod, model_cls)[model_member]

    mock_client = MagicMock()
    mock_client.converse.return_value = _make_bedrock_response("content_filtered")

    with patch.dict(os.environ, {"AWS_ACCESS_KEY": "k", "AWS_SECRET_ACCESS_KEY": "s", "AWS_REGION": "us-east-1"}):
        # Reset module-level _client so our mock is used
        mod._client = mock_client
        parser = parser_cls()
        result = parser.fetch(sample_jpeg, model=model_enum)

    assert result["body"] is None
    assert result["filtered"] is True
    assert result["status"] == 200
    assert result["full"] is not None


@pytest.mark.parametrize("module,cls,model_cls,model_member", BEDROCK_PARSERS)
def test_bedrock_max_tokens(module, cls, model_cls, model_member, sample_jpeg):
    import importlib
    mod = importlib.import_module(module)
    parser_cls = getattr(mod, cls)
    model_enum = getattr(mod, model_cls)[model_member]

    mock_client = MagicMock()
    mock_client.converse.return_value = _make_bedrock_response("max_tokens", "partial text")

    with patch.dict(os.environ, {"AWS_ACCESS_KEY": "k", "AWS_SECRET_ACCESS_KEY": "s", "AWS_REGION": "us-east-1"}):
        mod._client = mock_client
        parser = parser_cls()
        result = parser.fetch(sample_jpeg, model=model_enum)

    assert result["body"] == "partial text"
    assert result["truncated"] is True
    assert result["status"] == 200
    assert "filtered" not in result


@pytest.mark.parametrize("module,cls,model_cls,model_member", BEDROCK_PARSERS)
def test_bedrock_normal_success(module, cls, model_cls, model_member, sample_jpeg):
    import importlib
    mod = importlib.import_module(module)
    parser_cls = getattr(mod, cls)
    model_enum = getattr(mod, model_cls)[model_member]

    mock_client = MagicMock()
    mock_client.converse.return_value = _make_bedrock_response("end_turn", "A painting of flowers.")

    with patch.dict(os.environ, {"AWS_ACCESS_KEY": "k", "AWS_SECRET_ACCESS_KEY": "s", "AWS_REGION": "us-east-1"}):
        mod._client = mock_client
        parser = parser_cls()
        result = parser.fetch(sample_jpeg, model=model_enum)

    assert result["body"] == "A painting of flowers."
    assert result["status"] == 200
    assert "filtered" not in result
    assert "truncated" not in result


# ── Google Gemini ─────────────────────────────────────────────────────────────

def _gemini_ok_response(finish_reason="STOP", text="A painting."):
    return {
        "candidates": [
            {
                "content": {"parts": [{"text": text}]},
                "finishReason": finish_reason
            }
        ]
    }


def _make_requests_mock(status_code, json_body):
    mock_resp = MagicMock()
    mock_resp.status_code = status_code
    mock_resp.json.return_value = json_body
    mock_resp.text = str(json_body)
    return mock_resp


@pytest.fixture()
def gemini_parser():
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
        from parsers.googlegemini import GoogleGemini, GoogleGeminiModel
        return GoogleGemini(), GoogleGeminiModel.FLASH_2_0


def test_gemini_normal_success(gemini_parser, sample_jpeg):
    parser, model = gemini_parser
    mock_resp = _make_requests_mock(200, _gemini_ok_response())
    with patch("requests.post", return_value=mock_resp):
        result = parser.fetch(sample_jpeg, model=model)
    assert result["body"] == "A painting."
    assert result["status"] == 200
    assert "filtered" not in result
    assert "truncated" not in result
    assert "content_policy_violation" not in result


def test_gemini_safety_filtered(gemini_parser, sample_jpeg):
    parser, model = gemini_parser
    mock_resp = _make_requests_mock(200, _gemini_ok_response(finish_reason="SAFETY"))
    with patch("requests.post", return_value=mock_resp):
        result = parser.fetch(sample_jpeg, model=model)
    assert result["body"] is None
    assert result["filtered"] is True
    assert result["status"] == 200


def test_gemini_max_tokens(gemini_parser, sample_jpeg):
    parser, model = gemini_parser
    mock_resp = _make_requests_mock(200, _gemini_ok_response(finish_reason="MAX_TOKENS", text="partial"))
    with patch("requests.post", return_value=mock_resp):
        result = parser.fetch(sample_jpeg, model=model)
    assert result["body"] == "partial"
    assert result["truncated"] is True
    assert "filtered" not in result


def test_gemini_prompt_blocked(gemini_parser, sample_jpeg):
    parser, model = gemini_parser
    body = {"promptFeedback": {"blockReason": "SAFETY"}, "candidates": []}
    mock_resp = _make_requests_mock(200, body)
    with patch("requests.post", return_value=mock_resp):
        result = parser.fetch(sample_jpeg, model=model)
    assert result["body"] is None
    assert result["content_policy_violation"] is True
    assert result["status"] == 200


def test_gemini_http_error(gemini_parser, sample_jpeg):
    parser, model = gemini_parser
    mock_resp = _make_requests_mock(429, {"error": {"message": "rate limited"}})
    with patch("requests.post", return_value=mock_resp):
        result = parser.fetch(sample_jpeg, model=model)
    assert result["body"] is None
    assert result["status"] == 429
    assert result["full"] is None


def test_gemini_empty_candidates(gemini_parser, sample_jpeg):
    parser, model = gemini_parser
    mock_resp = _make_requests_mock(200, {"candidates": []})
    with patch("requests.post", return_value=mock_resp):
        result = parser.fetch(sample_jpeg, model=model)
    assert result["body"] is None
    assert result["status"] == 200
    assert "description" in result
