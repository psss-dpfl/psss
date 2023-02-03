# logarithmic calculations 
import math
import numpy as np
from scipy import special

def _log_add(logx, logy):
    # Add two numbers in the log space
    # log(x+y)
    a, b = min(logx, logy), max(logx, logy)
    if a == -np.inf:  # adding 0
        return b
    # Use exp(a) + exp(b) = (exp(a - b) + 1) * exp(b)
    return math.log1p(math.exp(a - b)) + b  # log1p(x) = log(x + 1)


def _log_comb(n, k):
    # the logarithm (base e) of n choose k
    return (special.gammaln(n + 1) - special.gammaln(k + 1) - special.gammaln(n - k + 1))

