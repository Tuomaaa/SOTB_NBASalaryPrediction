"""CBA constraint features.

Only encodes cba_era. Leakage features (is_vet_min, is_mle_range,
is_rookie_scale, max_eligible_pct) were removed — derived from target.
"""

import pandas as pd
from config import CBA_NEW_ERA_SEASON


def add_cba_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add CBA era flag.

    Input needs: season.
    Adds: cba_era (0=pre-2023, 1=post-2023).
    """
    df = df.copy()
    df["cba_era"] = (df["season"] >= CBA_NEW_ERA_SEASON).astype(int)
    return df
