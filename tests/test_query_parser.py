from utils.query_parser import parse_query


def test_parse_behind_color():
    q = parse_query("What is the color of the object behind the red cylinder?")
    assert q["attribute"] == "color"
    assert q["relation"] == "behind"
    assert q["target"] is not None
    assert "red" in q["target"] or "cylinder" in q["target"]
