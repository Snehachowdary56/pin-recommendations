"""Generated sample data: users saving pins on a fictional discovery platform.

Not Pinterest data. Users have 2-3 interests out of 24 topics; pins belong to one topic, have a short text
title, and popularity follows a long tail. Some pins are brand new (cold start): they appear only in the test
period, so a model that learns only from the save graph cannot recommend them, and a content model can.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

TOPICS = {
    "recipes": "easy dinner pasta chicken recipe healthy meal prep soup",
    "baking": "cake cookies bread sourdough dessert chocolate frosting oven",
    "home_decor": "living room cozy decor shelf wall art rug lamp",
    "gardening": "garden plants vegetable raised bed flowers soil seeds",
    "travel": "travel itinerary beach city guide budget trip hotel",
    "hiking": "hiking trail mountain backpacking camping gear national park",
    "fashion": "outfit street style fall jacket boots casual look",
    "skincare": "skincare routine serum moisturizer glow acne sunscreen",
    "fitness": "workout home exercise strength cardio routine abs",
    "yoga": "yoga poses stretch flow beginner mindfulness breathing",
    "diy": "diy craft project upcycle wood handmade easy tutorial",
    "wedding": "wedding bouquet dress venue decor centerpiece invitation",
    "kids": "kids activities toddler craft learning play rainy day",
    "pets": "dog cat pet care puppy training treats toys",
    "photography": "photography camera tips portrait lighting editing lens",
    "art": "drawing painting watercolor sketch illustration art ideas",
    "tech": "desk setup gadgets productivity workspace keyboard monitor",
    "finance": "budget savings money tips planner debt investing",
    "cars": "car restoration classic engine detailing road trip",
    "books": "reading list books novels bookshelf book club cozy",
    "nails": "nail art manicure design gel polish ideas",
    "hair": "hairstyle braids curls haircut short hair tutorial",
    "architecture": "modern house architecture interior design minimal",
    "coffee": "coffee latte espresso cafe barista morning brew",
}
TOPIC_NAMES = list(TOPICS)


@dataclass
class Dataset:
    pins: pd.DataFrame          # pin_id, topic, title, is_new
    users: pd.DataFrame         # user_id, interests
    train: pd.DataFrame         # user_id, pin_id, t
    valid: pd.DataFrame         # held out from train period, for the ranker
    test: pd.DataFrame          # last saves per user (includes new pins)

    @property
    def n_users(self) -> int:
        return len(self.users)

    @property
    def n_pins(self) -> int:
        return len(self.pins)


def generate(n_users: int = 3000, n_pins: int = 6000, new_pin_share: float = 0.08, seed: int = 0) -> Dataset:
    rng = np.random.default_rng(seed)
    k = len(TOPIC_NAMES)
    pin_topic = rng.integers(0, k, n_pins)
    words = {t: TOPICS[t].split() for t in TOPIC_NAMES}
    filler = "best ideas guide inspiration simple quick favorite".split()
    titles = []
    for t in pin_topic:
        w = list(rng.choice(words[TOPIC_NAMES[t]], 3, replace=False)) + list(rng.choice(filler, 1))
        rng.shuffle(w)
        titles.append(" ".join(w))
    is_new = rng.random(n_pins) < new_pin_share
    # long-tail popularity within each topic
    pop = rng.pareto(1.2, n_pins) + 1
    pins = pd.DataFrame({"pin_id": np.arange(n_pins), "topic": [TOPIC_NAMES[t] for t in pin_topic],
                         "title": titles, "is_new": is_new, "quality": pop})

    by_topic_old = {t: np.where((pin_topic == t) & ~is_new)[0] for t in range(k)}
    by_topic_new = {t: np.where((pin_topic == t) & is_new)[0] for t in range(k)}
    rows, interests = [], []
    for u in range(n_users):
        m = rng.integers(2, 4)
        ints = rng.choice(k, m, replace=False)
        w = rng.dirichlet(np.ones(m) * 2)
        interests.append([TOPIC_NAMES[i] for i in ints])
        n_saves = int(rng.integers(20, 61))
        for s in range(n_saves):
            late = s >= int(n_saves * 0.8)
            if rng.random() < 0.08:          # occasional off-interest exploration
                t = rng.integers(0, k)
            else:
                t = ints[rng.choice(m, p=w)]
            pool = by_topic_old[t]
            if late and rng.random() < 0.15 and len(by_topic_new[t]):
                pool = by_topic_new[t]       # new pins only show up in the last saves
            p = pins["quality"].to_numpy()[pool]
            rows.append((u, int(rng.choice(pool, p=p / p.sum())), s))
    inter = pd.DataFrame(rows, columns=["user_id", "pin_id", "t"]).drop_duplicates(["user_id", "pin_id"])
    users = pd.DataFrame({"user_id": np.arange(n_users), "interests": interests})

    # temporal split per user: last 20% test, the 10% before that validation
    inter = inter.sort_values(["user_id", "t"])
    inter["rank"] = inter.groupby("user_id").cumcount()
    inter["n"] = inter.groupby("user_id")["pin_id"].transform("size")
    frac = inter["rank"] / inter["n"]
    test = inter[frac >= 0.8]
    valid = inter[(frac >= 0.7) & (frac < 0.8)]
    train = inter[frac < 0.7]
    new_ids = set(pins.loc[pins["is_new"], "pin_id"])
    train = train[~train["pin_id"].isin(new_ids)]
    valid = valid[~valid["pin_id"].isin(new_ids)]
    cols = ["user_id", "pin_id", "t"]
    return Dataset(pins, users, train[cols].reset_index(drop=True), valid[cols].reset_index(drop=True),
                   test[cols].reset_index(drop=True))
