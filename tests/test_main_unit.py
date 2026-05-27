import pytest
from main import parse_service_features, _validate_prompt, _make_annotation_fragment, _scale_bbox


# ── parse_service_features ────────────────────────────────────────────────────

def test_single_service_no_features():
    assert parse_service_features("imagga") == {"imagga": ["all"]}

def test_service_with_single_feature():
    assert parse_service_features("imagga:tags") == {"imagga": ["tags"]}

def test_service_with_pipe_features():
    assert parse_service_features("imagga:tags|faces") == {"imagga": ["tags", "faces"]}

def test_comma_separated_services():
    assert parse_service_features("imagga,gv") == {"imagga": ["all"], "gv": ["all"]}

def test_mixed_services():
    assert parse_service_features("imagga:tags,gv") == {"imagga": ["tags"], "gv": ["all"]}

def test_all_keyword():
    assert parse_service_features("all") == {"all": ["all"]}

def test_empty_feature_after_colon():
    assert parse_service_features("imagga:") == {"imagga": ["all"]}

def test_whitespace_trimmed():
    assert parse_service_features(" imagga : tags ") == {"imagga": ["tags"]}

def test_empty_parts_skipped():
    assert parse_service_features("imagga,,gv") == {"imagga": ["all"], "gv": ["all"]}

def test_custom_default_value():
    assert parse_service_features("imagga", default_value="labels") == {"imagga": ["labels"]}

def test_whitespace_only_features():
    assert parse_service_features("imagga:  ") == {"imagga": ["all"]}


# ── _validate_prompt ──────────────────────────────────────────────────────────

def test_valid_short_prompt():
    assert _validate_prompt("Describe this image.") == ("Describe this image.", None)

def test_prompt_exactly_500_chars():
    s = "a" * 500
    assert _validate_prompt(s) == (s, None)

def test_prompt_501_chars_rejected():
    result, err = _validate_prompt("a" * 501)
    assert result == ""
    assert "500" in err

def test_null_byte_stripped():
    assert _validate_prompt("hello\x00world") == ("helloworld", None)

def test_tabs_and_newlines_preserved():
    assert _validate_prompt("hello\tworld\nnext") == ("hello\tworld\nnext", None)

def test_empty_string():
    assert _validate_prompt("") == ("", None)

def test_unicode_preserved():
    assert _validate_prompt("日本語テスト") == ("日本語テスト", None)

def test_zero_width_space_stripped():
    # ​ is category Cf — stripped by the filter
    assert _validate_prompt("hello​world") == ("helloworld", None)

def test_carriage_return_preserved():
    assert _validate_prompt("line1\rline2") == ("line1\rline2", None)


# ── _make_annotation_fragment ─────────────────────────────────────────────────

def test_annotation_integer_coords():
    assert _make_annotation_fragment(0, 0, 100, 200) == "xywh=0,0,100,200"

def test_annotation_float_coords_truncated():
    assert _make_annotation_fragment(1.7, 2.9, 100.1, 200.8) == "xywh=1,2,100,200"

def test_annotation_large_values():
    assert _make_annotation_fragment(1000, 2000, 3000, 4000) == "xywh=1000,2000,3000,4000"

def test_annotation_zero_origin():
    assert _make_annotation_fragment(0, 0, 500, 500) == "xywh=0,0,500,500"


# ── _scale_bbox ───────────────────────────────────────────────────────────────

def test_scale_factor_one():
    assert _scale_bbox(10, 20, 30, 40, 1.0) == (10.0, 20.0, 30.0, 40.0)

def test_scale_factor_two():
    assert _scale_bbox(10, 20, 30, 40, 2.0) == (20.0, 40.0, 60.0, 80.0)

def test_scale_factor_half():
    assert _scale_bbox(10, 20, 30, 40, 0.5) == (5.0, 10.0, 15.0, 20.0)

def test_scale_then_annotate():
    # Standard pipeline: scale then fragment — verifies int truncation
    result = _make_annotation_fragment(*_scale_bbox(10, 20, 30, 40, 0.5))
    assert result == "xywh=5,10,15,20"
