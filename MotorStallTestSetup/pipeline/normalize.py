from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


class FeatureNormalizer:
    def __init__(self, feature_columns: list[str]):
        self.feature_columns = feature_columns
        self.scaler = StandardScaler()
        self._fitted = False

    def fit(self, df: pd.DataFrame) -> "FeatureNormalizer":
        self.scaler.fit(df[self.feature_columns].values)
        self._fitted = True
        return self

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self._fitted:
            raise RuntimeError("Normalizer must be fit before transform.")
        out = df.copy()
        out[self.feature_columns] = self.scaler.transform(df[self.feature_columns].values)
        return out

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        return self.fit(df).transform(df)

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(
            {"scaler": self.scaler, "feature_columns": self.feature_columns},
            path,
        )

    @classmethod
    def load(cls, path: str | Path) -> "FeatureNormalizer":
        payload = joblib.load(path)
        obj = cls(payload["feature_columns"])
        obj.scaler = payload["scaler"]
        obj._fitted = True
        return obj
