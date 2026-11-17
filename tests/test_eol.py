"""Tests for end-of-life model gating in process_image._is_eol.

The guard compares ISO date strings lexicographically against today, and uses
`<=`, so a model is retired *on* its eol_date rather than the day after. These
tests pin both sides of that boundary.
"""
import datetime
from unittest.mock import patch, MagicMock


VALID_URL = "https://images.example.com/iiif/item123"

TODAY = datetime.date.today()
YESTERDAY = (TODAY - datetime.timedelta(days=1)).isoformat()
TOMORROW = (TODAY + datetime.timedelta(days=1)).isoformat()


# ── helpers ───────────────────────────────────────────────────────────────────

def _model(name, eol_date):
    """A stand-in model enum. eol_date must be a real str/None, not a Mock,
    because _is_eol compares it against a str."""
    m = MagicMock()
    m.name = name
    m.eol_date = eol_date
    return m


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


def _fetch_model(model, *args, **kwargs):
    return model.name, {"status": 200, "body": "result", "runtime": 0.01, "annotationFragment": ""}


def _run(client, model, fetch=_fetch_model):
    """Drive /extract for a single model, returning (payload, fetch_mock)."""
    fetch_mock = MagicMock(side_effect=fetch)
    patches = [
        patch("main.iiif.IIIFImage", return_value=_mock_iiif_image()),
        patch("main.cache.get_image", return_value=_mock_cache()),
        patch("main.GENERIC_MODELS", [(model, MagicMock, "full")]),
        patch("main._fetch_model", fetch_mock),
    ]
    for p in patches:
        p.start()
    try:
        r = client.get(f"/extract?url={VALID_URL}&services={model.name}")
        assert r.status_code == 200
        return r.get_json(), fetch_mock
    finally:
        for p in patches:
            p.stop()


# ── the <= boundary ───────────────────────────────────────────────────────────

def test_eol_date_today_is_retired(client):
    """The boundary case: a model is EOL on its eol_date, not the day after."""
    model = _model("eol-today", TODAY.isoformat())
    data, fetch_mock = _run(client, model)

    assert "error" in data["eol-today"]
    assert fetch_mock.call_count == 0, "EOL model must not be dispatched"


def test_eol_date_tomorrow_still_runs(client):
    """One day short of EOL the model is still called normally."""
    model = _model("eol-tomorrow", TOMORROW)
    data, fetch_mock = _run(client, model)

    assert data["eol-tomorrow"]["body"] == "result"
    assert "error" not in data["eol-tomorrow"]
    assert fetch_mock.call_count == 1


def test_eol_date_in_past_is_retired(client):
    model = _model("eol-yesterday", YESTERDAY)
    data, fetch_mock = _run(client, model)

    assert "error" in data["eol-yesterday"]
    assert fetch_mock.call_count == 0


def test_no_eol_date_runs(client):
    """eol_date of None is falsy and short-circuits before the comparison."""
    model = _model("no-eol", None)
    data, fetch_mock = _run(client, model)

    assert data["no-eol"]["body"] == "result"
    assert fetch_mock.call_count == 1


# ── the error payload ─────────────────────────────────────────────────────────

def test_eol_error_message_names_model_and_date(client):
    model = _model("eol-today", TODAY.isoformat())
    data, _ = _run(client, model)

    error = data["eol-today"]["error"]
    assert "eol-today" in error
    assert TODAY.isoformat() in error


def test_eol_payload_replaces_result_entirely(client):
    """The EOL entry carries only the error — no stale body/status from a call."""
    model = _model("eol-today", TODAY.isoformat())
    data, _ = _run(client, model)

    assert data["eol-today"] == {
        "error": f"model eol-today reached end-of-life on {TODAY.isoformat()}"
    }


# ── declared eol_dates must be comparable as strings ──────────────────────────

def _declared_models(client):
    services = client.get("/list/services").get_json()["services"]
    return [m for group in services.values() for m in group]


def test_every_declared_eol_date_is_zero_padded_iso(client):
    """_is_eol compares strings, so an unpadded date like '2026-6-1' would sort
    wrong against today's isoformat and silently retire the model early."""
    bad = []
    for m in _declared_models(client):
        eol = m.get("eol_date")
        if eol is None:
            continue
        try:
            parsed = datetime.date.fromisoformat(eol)
        except (TypeError, ValueError):
            bad.append((m["name"], eol))
            continue
        if parsed.isoformat() != eol:
            bad.append((m["name"], eol))

    assert not bad, f"eol_date must be zero-padded YYYY-MM-DD: {bad}"


def test_list_services_returns_models(client):
    """Guards the test above from passing vacuously on an empty payload."""
    assert len(_declared_models(client)) > 0
