import numpy as np


def logdiff(x):

    return np.log(1 + x.pct_change())
