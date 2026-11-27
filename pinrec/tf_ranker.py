"""The same stage-2 ranker in TensorFlow / Keras, to check that results do not depend on the framework."""
from __future__ import annotations

import os

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")

import numpy as np
import pandas as pd

from .rank import FEATURES, top_from_scores


def fit(df: pd.DataFrame, labels: pd.DataFrame, epochs: int = 8, lr: float = 3e-3, seed: int = 0):
    import tensorflow as tf

    tf.keras.utils.set_random_seed(seed)
    y = df.merge(labels.assign(y=1.0)[["user_id", "pin_id", "y"]], how="left", on=["user_id", "pin_id"])["y"].fillna(0).to_numpy(np.float32)
    x = df[FEATURES].to_numpy(np.float32)
    norm = tf.keras.layers.Normalization()
    norm.adapt(x)
    model = tf.keras.Sequential([
        tf.keras.Input(shape=(len(FEATURES),)), norm,
        tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dense(32, activation="relu"), tf.keras.layers.Dense(1),
    ])
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss=tf.keras.losses.BinaryCrossentropy(from_logits=True))
    pos_w = (1 - y.mean()) / max(y.mean(), 1e-6)
    model.fit(x, y, batch_size=8192, epochs=epochs, class_weight={0: 1.0, 1: float(pos_w)}, verbose=0)
    return model


def rank(model, df: pd.DataFrame, n_users: int, k: int, fresh_slots: int = 0) -> np.ndarray:
    s = model.predict(df[FEATURES].to_numpy(np.float32), batch_size=65536, verbose=0).ravel()
    return top_from_scores(df, s, n_users, k, fresh_slots)
