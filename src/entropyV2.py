import numpy as np


def entropy_from_reserve(r):

    s = r.sum()

    if s == 0:
        return 0

    p = r / s

    return -np.sum(p * np.log2(p + 1e-12))


def compute_entropy_v2(df):

    if 'reserve0' in df.columns and 'reserve1' in df.columns:
        df["entropy"] = df.apply(
            lambda x: entropy_from_reserve(
                np.array([float(x.reserve0), float(x.reserve1)], dtype=float)
            ),
            axis=1
        )
    elif 'token0Price' in df.columns and 'token1Price' in df.columns:
        df["entropy"] = df.apply(
            lambda x: entropy_from_reserve(
                np.array([float(x.token0Price), float(x.token1Price)], dtype=float)
            ),
            axis=1
        )

    return df
