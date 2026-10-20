"""Combo Lab pure helpers: enumeration, ranking and search summary.

The server does the expensive rollout; these tests pin the rules that decide
which combo wins and how the search is reported, so a UI change cannot silently
turn "ranked on validation" into "ranked on test".
"""
from combo_lab import (MAX_COMBOS, combos_for, normalise_pool, rank,
                       summarise)


def _row(combo, ret, bh, **kw):
    return {"combo": list(combo), "return_pct": ret, "bh_return_pct": bh,
            "excess_pct": round(ret - bh, 2), **kw}


def test_pool_is_uppercased_deduplicated_and_ordered():
    assert normalise_pool([" msft ", "KO", "msft", "", "ko"]) == ["MSFT", "KO"]


def test_combos_are_all_subsets_in_lexicographic_order():
    combos = combos_for(["A", "B", "C", "D"], 3)
    assert combos == [("A", "B", "C"), ("A", "B", "D"),
                      ("A", "C", "D"), ("B", "C", "D")]
    assert len(combos_for(list("ABCDEFGHIJKL"), 3)) == 220
    assert len(combos_for(list("ABCDEFGHIJKL"), 4)) == 495


def test_combos_empty_when_size_exceeds_pool():
    assert combos_for(["A", "B"], 3) == []
    assert combos_for(["A", "B"], 1) == [("A",), ("B",)]


def test_rank_uses_validation_return_then_simpler_basket():
    a = _row(("A", "B"), 30.0, 10.0)
    b = _row(("C", "D"), 22.0, 2.0)
    bad = {"combo": ["X", "Y"], "error": "no data"}
    # The highest validation return wins even with the smallest excess.
    c = _row(("E", "F"), 300.0, 290.0)
    ranked = rank([c, a, b, bad])
    assert [r["combo"] for r in ranked] == [["E", "F"], ["A", "B"], ["C", "D"]]


def test_rank_tie_break_prefers_the_simpler_basket_then_excess():
    """Same fly return: a 2-name basket beats a 3-name one, so adding an asset
    that the fly ignores cannot win on a weaker benchmark. Same size: the
    higher validation excess wins."""
    pair = _row(("A", "B"), 50.0, 10.0)
    triple = _row(("A", "B", "C"), 50.0, 5.0)    # excess +45
    triple2 = _row(("A", "B", "D"), 50.0, 20.0)  # excess +30
    ranked = rank([triple, triple2, pair])
    assert ranked[0]["combo"] == ["A", "B"]
    assert ranked[1]["combo"] == ["A", "B", "C"]
    assert ranked[2]["combo"] == ["A", "B", "D"]


def test_rank_does_not_use_test_fields():
    """Only validation fields exist at ranking time; a test block is ignored."""
    a = _row(("A", "B"), 10.0, 1.0)
    b = _row(("C", "D"), 11.0, 1.0)
    # A has the better *test* result, B the better validation excess.
    a["test"] = {"return_pct": 100.0, "bh_return_pct": 0.0, "excess_pct": 100.0}
    b["test"] = {"return_pct": 1.0, "bh_return_pct": 0.0, "excess_pct": 1.0}
    assert rank([a, b])[0] is b


def test_summary_counts_the_distribution_not_just_the_winner():
    rows = [_row(("A", "B"), 30.0, 10.0),     # +20
            _row(("C", "D"), 5.0, 8.0),       # -3
            _row(("E", "F"), 12.0, 9.0)]      # +3
    s = summarise(rows, total=3)
    assert s == {
        "evaluated": 3, "total": 3, "beat_bh": 2,
        "median_excess_pct": 3.0, "mean_excess_pct": 6.67,
        "median_return_pct": 12.0,
    }


def test_summary_with_no_results_is_empty_not_an_error():
    s = summarise([{"combo": ["A", "B"], "error": "boom"}], total=1)
    assert s["evaluated"] == 0 and s["beat_bh"] == 0
    assert s["median_excess_pct"] is None


def test_limit_is_large_enough_for_the_default_universe():
    # The shipped 12-ticker universe must fit triples but not quadruples.
    assert len(combos_for(list("ABCDEFGHIJKL"), 3)) < MAX_COMBOS
    assert len(combos_for(list("ABCDEFGHIJKL"), 4)) > MAX_COMBOS
