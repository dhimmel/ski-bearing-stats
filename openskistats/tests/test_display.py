from openskistats.display import country_code_to_emoji


def test_country_code_to_emoji() -> None:
    assert country_code_to_emoji("US") == "🇺🇸"
    assert country_code_to_emoji("FR") == "🇫🇷"
