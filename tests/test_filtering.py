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


AWS_ENV = {"AWS_ACCESS_KEY": "k", "AWS_SECRET_ACCESS_KEY": "s", "AWS_REGION": "us-east-1"}


def _anthropic_fetch(content, stop_reason, sample_jpeg):
    import parsers.awsanthropic as mod

    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "stopReason": stop_reason,
        "output": {"message": {"content": content}},
    }

    with patch.dict(os.environ, AWS_ENV):
        mod._client = mock_client
        return mod.AWSAnthropic().fetch(sample_jpeg, model=mod.AnthropicModel.CLAUDE_5_5_SONNET)


def test_anthropic_reasoning_block_before_text(sample_jpeg):
    """Reasoning models put a reasoningContent block ahead of the text block."""
    result = _anthropic_fetch(
        [
            {"SDK_UNKNOWN_MEMBER": {"name": "reasoningContent"}},
            {"text": "A silkscreen portrait."},
        ],
        "end_turn",
        sample_jpeg,
    )

    assert result["body"] == "A silkscreen portrait."
    assert result["status"] == 200
    assert "description" not in result


def test_anthropic_reasoning_disabled_is_sent(sample_jpeg):
    """Models carrying a passthrough send it; Sonnet 5.5 turns reasoning off."""
    import parsers.awsanthropic as mod

    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "stopReason": "end_turn",
        "output": {"message": {"content": [{"text": "A portrait."}]}},
    }

    with patch.dict(os.environ, AWS_ENV):
        mod._client = mock_client
        mod.AWSAnthropic().fetch(sample_jpeg, model=mod.AnthropicModel.CLAUDE_5_5_SONNET)

    sent = mock_client.converse.call_args.kwargs
    assert sent["additionalModelRequestFields"] == {"thinking": {"type": "between_tools"}}


def test_anthropic_no_passthrough_by_default(sample_jpeg):
    """Models without a passthrough send no additionalModelRequestFields at all.

    An empty dict is not equivalent - Bedrock validates what it receives, and the
    value valid for one model is rejected by another.
    """
    import parsers.awsanthropic as mod

    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "stopReason": "end_turn",
        "output": {"message": {"content": [{"text": "A portrait."}]}},
    }

    with patch.dict(os.environ, AWS_ENV):
        mod._client = mock_client
        mod.AWSAnthropic().fetch(sample_jpeg, model=mod.AnthropicModel.CLAUDE_4_5_SONNET)

    assert "additionalModelRequestFields" not in mock_client.converse.call_args.kwargs
    assert mod.AnthropicModel.CLAUDE_4_5_SONNET.additional_request_fields is None


def test_moonshot_reasoning_block_before_text(sample_jpeg):
    """Kimi K3 reasons and cannot be told not to, so text is never content[0]."""
    import parsers.awsmoonshot as mod

    mock_client = MagicMock()
    mock_client.converse.return_value = {
        "stopReason": "end_turn",
        "output": {"message": {"content": [
            {"SDK_UNKNOWN_MEMBER": {"name": "reasoningContent"}},
            {"text": "A flat graphic."},
        ]}},
    }

    with patch.dict(os.environ, AWS_ENV):
        mod._client = mock_client
        result = mod.AWSMoonshot().fetch(sample_jpeg, model=mod.MoonshotModel.KIMI_K_3)

    assert result["body"] == "A flat graphic."
    assert result["status"] == 200


def test_moonshot_k3_uses_inference_profile(sample_jpeg):
    """K3 has no on-demand throughput - the ID must name an inference profile."""
    import parsers.awsmoonshot as mod

    model_id = mod.MoonshotModel.KIMI_K_3.model_id
    assert model_id.split(".")[0] in ("us", "eu", "apac", "global"), model_id
    # K3 rejects both sampling fields outright
    assert "temperature" not in mod.MoonshotModel.KIMI_K_3.inference_config
    assert "topP" not in mod.MoonshotModel.KIMI_K_3.inference_config


def test_anthropic_no_text_block(sample_jpeg):
    """A response truncated mid-reasoning carries no text block at all."""
    result = _anthropic_fetch(
        [{"SDK_UNKNOWN_MEMBER": {"name": "reasoningContent"}}],
        "max_tokens",
        sample_jpeg,
    )

    assert result["body"] is None
    assert result["truncated"] is True
    assert result["status"] == 200
    assert "description" in result


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


def _sent_body(mock_post):
    """The JSON payload handed to requests.post."""
    return mock_post.call_args.kwargs["json"]


def _gemini_models():
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
        from parsers.googlegemini import GoogleGeminiModel
        return list(GoogleGeminiModel)


@pytest.mark.parametrize("model", _gemini_models(), ids=lambda m: m.name)
def test_gemini_thinking_config_sent(model, sample_jpeg):
    """Whatever a model declares is what reaches the wire, and a model that
    declares nothing sends no generationConfig at all. Derived from the enum so
    retuning a level is not a test change — the rules below are what's pinned."""
    with patch.dict(os.environ, {"GOOGLE_API_KEY": "test-key"}):
        from parsers.googlegemini import GoogleGemini
        mock_resp = _make_requests_mock(200, _gemini_ok_response())
        with patch("requests.post", return_value=mock_resp) as mock_post:
            GoogleGemini().fetch(sample_jpeg, model=model)

    body = _sent_body(mock_post)
    if model.thinking_config is None:
        assert "generationConfig" not in body
    else:
        assert body["generationConfig"]["thinkingConfig"] == model.thinking_config


@pytest.mark.parametrize("model", _gemini_models(), ids=lambda m: m.name)
def test_gemini_thinking_field_matches_model_generation(model):
    """2.5 takes thinkingBudget, 3.x takes thinkingLevel, 2.0 does not reason.
    Using the wrong field for the generation is a 400 from the API."""
    cfg = model.thinking_config or {}
    assert not ("thinkingBudget" in cfg and "thinkingLevel" in cfg), "fields are mutually exclusive"

    if model.model_id.startswith("gemini-2.0"):
        assert model.thinking_config is None
    elif model.model_id.startswith("gemini-2.5"):
        assert set(cfg) == {"thinkingBudget"}
    else:
        assert set(cfg) == {"thinkingLevel"}


@pytest.mark.parametrize("model", _gemini_models(), ids=lambda m: m.name)
def test_gemini_thinking_level_value_is_supported(model):
    """"minimal" 400s on 3.x Flash and 3 Pro — only Flash-Lite may use it."""
    level = (model.thinking_config or {}).get("thinkingLevel")
    if level is None:
        return
    assert level in {"minimal", "low", "medium", "high"}
    if level == "minimal":
        assert "flash-lite" in model.model_id


def test_gemini_text_part_with_thought_signature(gemini_parser, sample_jpeg):
    """Gemini 3 attaches a thoughtSignature to the part carrying the answer."""
    parser, model = gemini_parser
    body = {
        "candidates": [{
            "content": {"parts": [
                {"text": "A painting.", "thoughtSignature": "Cr4BAdHtim8abc=="}
            ]},
            "finishReason": "STOP"
        }]
    }
    with patch("requests.post", return_value=_make_requests_mock(200, body)):
        result = parser.fetch(sample_jpeg, model=model)
    assert result["body"] == "A painting."
    assert result["status"] == 200


def test_gemini_thought_part_before_text(gemini_parser, sample_jpeg):
    """A thought part ahead of the answer must not be mistaken for the answer."""
    parser, model = gemini_parser
    body = {
        "candidates": [{
            "content": {"parts": [
                {"text": "First I consider the composition...", "thought": True},
                {"text": "A painting."}
            ]},
            "finishReason": "STOP"
        }]
    }
    with patch("requests.post", return_value=_make_requests_mock(200, body)):
        result = parser.fetch(sample_jpeg, model=model)
    assert result["body"] == "A painting."


def test_gemini_multiple_text_parts_joined(gemini_parser, sample_jpeg):
    parser, model = gemini_parser
    body = {
        "candidates": [{
            "content": {"parts": [{"text": "A painting"}, {"text": " of a ship."}]},
            "finishReason": "STOP"
        }]
    }
    with patch("requests.post", return_value=_make_requests_mock(200, body)):
        result = parser.fetch(sample_jpeg, model=model)
    assert result["body"] == "A painting of a ship."


def test_gemini_truncated_mid_thought_has_no_text_part(gemini_parser, sample_jpeg):
    """Cut off while thinking: no text part at all. Must report a 200 with a
    null body, not a 500 from a KeyError."""
    parser, model = gemini_parser
    body = {
        "candidates": [{
            "content": {"parts": [{"text": "Considering...", "thought": True}]},
            "finishReason": "MAX_TOKENS"
        }]
    }
    with patch("requests.post", return_value=_make_requests_mock(200, body)):
        result = parser.fetch(sample_jpeg, model=model)
    assert result["status"] == 200
    assert result["body"] is None
    assert result["truncated"] is True
    assert "no text part" in result["description"]


def test_gemini_candidate_without_content(gemini_parser, sample_jpeg):
    """A candidate carrying no content block at all must not raise."""
    parser, model = gemini_parser
    body = {"candidates": [{"finishReason": "MAX_TOKENS"}]}
    with patch("requests.post", return_value=_make_requests_mock(200, body)):
        result = parser.fetch(sample_jpeg, model=model)
    assert result["status"] == 200
    assert result["body"] is None


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
