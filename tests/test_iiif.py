import pytest
import requests
from parsers.iiif import IIIFImage

BASE_URI = "https://images.example.com/iiif/item123"
INFO_URL = f"{BASE_URI}/info.json"

INFO_PAYLOAD = {
    "@context": "http://iiif.io/api/image/2/context.json",
    "@id": BASE_URI,
    "protocol": "http://iiif.io/api/image",
    "width": 5000,
    "height": 4000,
    "profile": ["http://iiif.io/api/image/2/level2.json"],
}


# ── Validation / error cases ──────────────────────────────────────────────────

def test_empty_uri_raises():
    with pytest.raises(ValueError):
        IIIFImage("")

def test_non_url_string_raises():
    with pytest.raises(ValueError):
        IIIFImage("just-a-string")

def test_valid_info_json_is_valid(requests_mock):
    requests_mock.get(INFO_URL, json=INFO_PAYLOAD)
    img = IIIFImage(BASE_URI)
    assert img.is_valid() is True
    assert img.get_status() == "ok"

def test_404_is_invalid(requests_mock):
    requests_mock.get(INFO_URL, status_code=404)
    img = IIIFImage(BASE_URI)
    assert img.is_valid() is False
    assert img.get_status() == "bad"

def test_timeout_is_invalid(requests_mock):
    requests_mock.get(INFO_URL, exc=requests.exceptions.Timeout)
    img = IIIFImage(BASE_URI)
    assert img.is_valid() is False
    assert img.get_status() == "bad"

def test_bad_json_is_invalid(requests_mock):
    requests_mock.get(INFO_URL, text="not valid json", status_code=200)
    img = IIIFImage(BASE_URI)
    assert img.is_valid() is False
    assert img.get_status() == "bad"


# ── URL construction ──────────────────────────────────────────────────────────

@pytest.fixture()
def iiif(requests_mock):
    requests_mock.get(INFO_URL, json=INFO_PAYLOAD)
    return IIIFImage(BASE_URI)

def test_get_base_uri(iiif):
    assert iiif.get_base_uri() == BASE_URI

def test_trailing_slash_stripped(requests_mock):
    requests_mock.get(INFO_URL, json=INFO_PAYLOAD)
    img = IIIFImage(BASE_URI + "/")
    assert not img.get_base_uri().endswith("/")

def test_get_full_image_url(iiif):
    assert iiif.get_full_image_url() == f"{BASE_URI}/full/full/0/default.jpg"

def test_get_scaled_image_url(iiif):
    assert iiif.get_scaled_image_url("!150,150") == f"{BASE_URI}/full/!150,150/0/default.jpg"

def test_get_scaled_image_url_max(iiif):
    assert iiif.get_scaled_image_url("max") == f"{BASE_URI}/full/max/0/default.jpg"

def test_get_scaled_empty_raises(iiif):
    with pytest.raises(ValueError):
        iiif.get_scaled_image_url("")

def test_get_fragment_url(iiif):
    assert iiif.get_fragment_image_url(10, 20, 100, 200) == f"{BASE_URI}/10,20,100,200/full/0/default.jpg"

def test_get_fragment_zero_origin(iiif):
    assert iiif.get_fragment_image_url(0, 0, 500, 500) == f"{BASE_URI}/0,0,500,500/full/0/default.jpg"

def test_fragment_negative_coord_raises(iiif):
    with pytest.raises(ValueError):
        iiif.get_fragment_image_url(-1, 0, 100, 100)

def test_fragment_zero_width_raises(iiif):
    with pytest.raises(ValueError):
        iiif.get_fragment_image_url(0, 0, 0, 100)

def test_fragment_zero_height_raises(iiif):
    with pytest.raises(ValueError):
        iiif.get_fragment_image_url(0, 0, 100, 0)


# ── info data ─────────────────────────────────────────────────────────────────

def test_info_width_height(iiif):
    assert iiif.info["width"] == 5000
    assert iiif.info["height"] == 4000

def test_fetch_returns_info(iiif):
    assert iiif.fetch() == iiif.info


# ── Known URL ID extraction ───────────────────────────────────────────────────

def test_harvard_ids_mapping(requests_mock):
    uri = "https://ids.lib.harvard.edu/ids/12345"
    requests_mock.get(f"{uri}/info.json", json=INFO_PAYLOAD)
    img = IIIFImage(uri)
    assert img.get_id() == "12345"

def test_harvard_mps_mapping(requests_mock):
    uri = "https://ids.lib.harvard.edu/mps/67890"
    requests_mock.get(f"{uri}/info.json", json=INFO_PAYLOAD)
    img = IIIFImage(uri)
    assert img.get_id() == "67890"

def test_unknown_url_returns_minus_one(requests_mock):
    requests_mock.get(INFO_URL, json=INFO_PAYLOAD)
    img = IIIFImage(BASE_URI)
    assert img.get_id() == -1
