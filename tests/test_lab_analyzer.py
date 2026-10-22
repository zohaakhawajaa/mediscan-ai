from modules.lab_analyzer import analyze_entries, range_options_for


def test_hemoglobin_low_female():
    result = analyze_entries([
        {"test": "hemoglobin", "value": "10.8", "range": "adult_female"}])
    assert result["results"][0]["flag"] == "low"
    assert result["results"][0]["unit"] == "g/dL"


def test_hemoglobin_normal_male():
    result = analyze_entries([
        {"test": "hemoglobin", "value": "15.0", "range": "adult_male"}])
    assert result["results"][0]["flag"] == "normal"


def test_glucose_high_fasting():
    result = analyze_entries([
        {"test": "glucose", "value": "130", "range": "fasting_adult"}])
    assert result["results"][0]["flag"] == "high"


def test_wbc_high():
    result = analyze_entries([
        {"test": "wbc", "value": "15.2", "range": "adult"}])
    assert result["results"][0]["flag"] == "high"


def test_platelets_low():
    result = analyze_entries([
        {"test": "platelets", "value": "120", "range": "adult"}])
    assert result["results"][0]["flag"] == "low"


def test_unknown_test_reported_not_fabricated():
    result = analyze_entries([
        {"test": "troponin", "value": "0.1", "range": "adult"}])
    assert result["results"] == []
    assert result["errors"]


def test_invalid_value_reported():
    result = analyze_entries([
        {"test": "glucose", "value": "abc", "range": "fasting_adult"}])
    assert result["results"] == []
    assert result["errors"]


def test_unknown_range_reported():
    result = analyze_entries([
        {"test": "hemoglobin", "value": "14", "range": "pregnant"}])
    assert result["results"] == []
    assert any("unknown reference range" in e.lower() for e in result["errors"])


def test_range_options():
    assert "adult_female" in range_options_for("hemoglobin")
    assert range_options_for("nonexistent") == []
