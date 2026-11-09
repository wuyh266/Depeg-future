import numpy as np


def compute_netflow(swaps, base="USDC"):

    swaps = swaps.copy()

    swaps["flow"] = 0.0

    if 'amount0In' in swaps.columns and 'amount0Out' in swaps.columns:
        swaps["flow"] += swaps["amount0In"].astype(float)
        swaps["flow"] -= swaps["amount0Out"].astype(float)
    elif 'amount0' in swaps.columns and 'amount1' in swaps.columns:
        swaps["flow"] += swaps["amount0"].astype(float)
        swaps["flow"] -= swaps["amount1"].astype(float)

    return (
        swaps
        .groupby("hour")["flow"]
        .sum()
        .reset_index()
    )
