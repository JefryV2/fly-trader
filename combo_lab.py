"""
combo_lab.py
============
Pure helpers for the dashboard's **Combo Lab**: enumerate k-stock subsets of a
universe, rank the evaluated results, and summarise the search.

This module is deliberately free of torch / pandas / file I/O so the ranking
rules are unit-testable in isolation. The server does the expensive part (build
a ``PortfolioEnv`` per combo, run the deployed portfolio fly, compare the same
basket's buy & hold) and hands the result dicts here.

Honesty contract
----------------
Combos are ranked on the **validation** split only. The winner is then scored
once on the **test** split and reported separately: choosing the best of N
combos on validation is a search, so the winner's validation number is
optimistic by construction. The test number is the honest one.
"""
from __future__ import annotations

from itertools import combinations
from statistics import mean, median

# One dashboard request should finish in a couple of minutes. C(12, 3) = 220;
# C(12, 4) = 495 and C(14, 3) = 364 are rejected with a message that tells the
# user to shrink the pool or the subset size.
MAX_COMBOS = 240


def normalise_pool(pool) -> list[str]:
    """Uppercase, de-duplicate and preserve order."""
    out: list[str] = []
    for t in pool or []:
        t = str(t).strip().upper()
        if t and t not in out:
            out.append(t)
    return out


def combos_for(pool, size: int) -> list[tuple[str, ...]]:
    """Every ``size``-ticker combo of ``pool``, in lexicographic order."""
    pool = normalise_pool(pool)
    size = int(size)
    if size < 1 or size > len(pool):
        return []
    return [tuple(c) for c in combinations(pool, size)]


def rank(results) -> list[dict]:
    """Best first: the fly's **validation return**, then the simpler basket, then
    excess vs that basket's buy & hold.

    Why return and not excess? Ranking by excess rewards a combo whose
    *benchmark* is weak, not one where the fly traded better: adding a dragging
    asset can lower buy & hold while the fly's own P&L is unchanged, and the
    combo then "wins" for the wrong reason. The fly's return is what the user
    asked to maximise; excess is still shown on every row, and the winner's test
    row is the honest out-of-sample check.
    """
    ok = [r for r in results if not r.get("error")]
    ok.sort(key=lambda r: (float(r.get("return_pct", 0.0)),
                           -len(r.get("combo", [])),
                           float(r.get("excess_pct", 0.0))),
            reverse=True)
    return ok


def summarise(results, total: int) -> dict:
    """Distribution of the search, so the top row is not shown alone.

    A long list where the median combo *loses* to buy & hold is evidence that
    the winner is a selection artefact, not a property of the fly.
    """
    ok = [r for r in results if not r.get("error")]
    total = int(total)
    if not ok:
        return {"evaluated": 0, "total": total, "beat_bh": 0,
                "failed": total,
                "median_excess_pct": None, "mean_excess_pct": None,
                "median_return_pct": None}
    ex = [float(r.get("excess_pct", 0.0)) for r in ok]
    ret = [float(r.get("return_pct", 0.0)) for r in ok]
    return {
        "evaluated": len(ok),
        "total": total,
        "failed": total - len(ok),
        "beat_bh": sum(1 for e in ex if e > 0),
        "median_excess_pct": round(median(ex), 2),
        "mean_excess_pct": round(mean(ex), 2),
        "median_return_pct": round(median(ret), 2),
    }
