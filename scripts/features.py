"""Feature engineering shared by scoring and training, so both see the same inputs."""
import numpy as np
import pandas as pd

FEATURES = ["log_runoff", "twi_z", "low_elev_share", "built_share"]


def scs_runoff_mm(p_mm, cn):
    """SCS Curve Number direct runoff depth (mm) for rainfall depth p_mm."""
    p = np.asarray(p_mm, dtype=float)
    cn = np.clip(np.asarray(cn, dtype=float), 30, 99.5)
    s = 25400.0 / cn - 254.0
    ia = 0.2 * s
    q = np.where(p > ia, (p - ia) ** 2 / (p - ia + s), 0.0)
    return q


def build_features(df, static_ref):
    """df needs p24 (mm), cn_mean, twi_mean, low_elev_share, built_share.

    static_ref is the full ward table; TWI is standardised against it so the
    same ward gets the same value at training and scoring time.
    """
    twi_mu = float(static_ref["twi_mean"].mean())
    twi_sd = float(static_ref["twi_mean"].std()) or 1.0
    cn = df["cn_mean"].fillna(static_ref["cn_mean"].median())
    out = pd.DataFrame(index=df.index)
    out["runoff_mm"] = scs_runoff_mm(df["p24"].fillna(0), cn)
    out["log_runoff"] = np.log1p(out["runoff_mm"])
    out["twi_z"] = ((df["twi_mean"] - twi_mu) / twi_sd).fillna(0)
    out["low_elev_share"] = df["low_elev_share"].fillna(0).clip(0, 1)
    out["built_share"] = df["built_share"].fillna(0).clip(0, 1)
    return out
