"""Age-based features for the valuation model.

Adds age, age_squared, and experience_years. The quadratic term captures
the nonlinear decline after peak years (~27-28).
"""

import pandas as pd


def add_age_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add age-derived features.

    Input must have column `age` (float, at time of season).
    Adds: age_squared, years_from_peak (signed distance from age 27).
    """
    df = df.copy()

    if "age" not in df.columns:
        return df

    df["age_squared"] = df["age"] ** 2
    df["years_from_peak"] = df["age"] - 27.0

    return df


def add_experience(df: pd.DataFrame) -> pd.DataFrame:
    """Estimate experience_years from earliest season in the dataset.

    Requires columns: player_name_norm, season.
    This is an approximation — true rookie year may predate our data window.
    """
    df = df.copy()

    if "player_name_norm" not in df.columns:
        return df

    first_season = df.groupby("player_name_norm")["season"].transform("min")
    df["experience_years"] = df["season"] - first_season

    return df
