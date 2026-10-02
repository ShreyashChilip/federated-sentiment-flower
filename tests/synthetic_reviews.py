"""SYNTHETIC review file in the Amazon Reviews 2023 format, for tests only.

Nothing generated here is data about the world and no number computed from it
may be reported. The generator exists so that the natural-client code path can
be tested without downloading a multi-gigabyte category file.

Construction (so that tests have a known ground truth):
  * users belong to a "generous" or a "harsh" group, which shifts their ratings;
  * the words ``excellent`` / ``awful`` follow the rating of the individual
    review (a genuine within-client sentiment signal);
  * the words ``sigplus`` / ``sigminus`` are habits of the generous / harsh
    group and are unrelated to the rating inside a user (association that
    exists only across clients);
  * ``filler*`` words are noise.
The file also contains a few deliberately invalid lines.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np


def write_reviews(path: Path, num_users: int = 60, num_items: int = 25, seed: int = 0, invalid_lines: bool = True) -> dict:
    rng = np.random.default_rng(seed)
    lines, valid = [], 0
    for u in range(num_users):
        generous = u % 2 == 0
        n_reviews = int(rng.integers(3, 40))
        for _ in range(n_reviews):
            base = rng.choice([4, 5, 5, 3]) if generous else rng.choice([1, 2, 2, 3])
            rating = int(np.clip(base + rng.choice([-1, 0, 0, 1]), 1, 5))
            words = [f"filler{rng.integers(0, 30)}" for _ in range(6)]
            if rng.random() < 0.6:
                # depends on the rating relative to this user's own habit
                words.append("excellent" if rating >= base else "awful")
            if rng.random() < 0.7:
                words.append("sigplus" if generous else "sigminus")
            rng.shuffle(words)
            lines.append(json.dumps({
                "rating": float(rating), "title": f"title{rng.integers(0, 5)}", "text": " ".join(words), "images": [],
                "asin": f"A{rng.integers(0, num_items):04d}", "parent_asin": f"P{rng.integers(0, num_items):04d}",
                "user_id": f"U{u:04d}", "timestamp": int(1_600_000_000_000 + valid), "helpful_vote": 0,
                "verified_purchase": True,
            }))
            valid += 1
    extra = {"bad_rating": 0, "empty_text": 0, "missing_id": 0, "duplicate": 0, "bad_json": 0}
    if invalid_lines:
        lines.append(json.dumps({"rating": 0.0, "title": "t", "text": "zero stars", "parent_asin": "P0001", "user_id": "U0001"}))
        lines.append(json.dumps({"rating": 5.0, "title": "t", "text": "   ", "parent_asin": "P0001", "user_id": "U0001"}))
        lines.append(json.dumps({"rating": 5.0, "title": "t", "text": "no user", "parent_asin": "P0001", "user_id": ""}))
        lines.append(lines[0])
        lines.append("{not json")
        extra = {"bad_rating": 1, "empty_text": 1, "missing_id": 1, "duplicate": 1, "bad_json": 1}
    Path(path).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return {"valid": valid, **extra}
