import numpy as np

from log_maths import _log_add
from log_maths import _log_comb

import argparse

def rdp_sk(alpha, mu_multi, gamma, d):
    l2 = np.sqrt(gamma**2+d)
    l1 = min(l2**2, np.sqrt(d)*l2)
    mu = mu_multi*(gamma**2)
    sum = alpha*l2*l2 / 4 / mu + min(((2*alpha-1)*l2*l2+6*l1)/16/mu/mu, 3*l1/4/mu)
    l2 = np.power(np.sqrt(gamma**2+d),3)
    l1 = min(l2**2, np.sqrt(d)*l2)
    mu = mu_multi*(gamma**6)
    sum = sum + alpha*l2*l2 / 4 / mu + min(((2*alpha-1)*l2*l2+6*l1)/16/mu/mu, 3*l1/4/mu)
    l2 = gamma*np.sqrt(gamma**2+d)
    l1 = min(l2**2, np.sqrt(d)*l2)
    mu = mu_multi*(gamma**4)
    sum = sum + alpha*l2*l2 / 4 / mu + min(((2*alpha-1)*l2*l2+6*l1)/16/mu/mu, 3*l1/4/mu)
    return sum

def rdp_sample_sk(q, alpha, mu_multi, gamma, d):
    current_log_sum = (alpha-1)*np.log(1-q) + np.log(alpha*q-q+1)
    for k in range(2,alpha+1):
        current_term =  _log_comb(alpha, k)+ (alpha-k)*np.log(1-q) + k*np.log(q) + \
            (k-1)*rdp_sk(k, mu_multi, gamma, d)
        current_log_sum = _log_add(current_log_sum, current_term)
    return current_log_sum / (alpha-1)


"""
Example:
  python psss_lr_analysis.py --epochs=30 --q=0.001 --gamma=100 --d=51 --mu_multiplier=0.2639
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # DATA
    parser.add_argument("--q", help="sampling_rate per batch", type=float, default=0.1)
    parser.add_argument("--d", help="dimension of x", type=float)

    # TRAIN
    parser.add_argument("--epochs", help="", type=int)
    parser.add_argument("--gamma", help="scale parameter", type=float, default=1)

    # NOISE
    parser.add_argument("--mu_multiplier", help="scale of overall skellam multiplier", type=float, default=0.01)

    # privacy meter
    parser.add_argument("--delta", help="delta in approximate DP", type=float, default=0.00001)

    args = parser.parse_args()

    print(args)

    q = args.q
    d = args.d
    epochs = args.epochs
    gamma = args.gamma
    mu_multiplier = args.mu_multiplier
    delta = args.delta
    
    rounds = int(1 / q) * epochs

    best_alpha = 0
    best_eps = np.inf
    alpha_candidates = np.array(range(2,200))
    for alpha in alpha_candidates:
        #print(alpha)
        current_eps = np.inf
        if q == 1:
            current_eps = rdp_sk(alpha, alpha, mu_multiplier, gamma, d)
        else:
            current_eps = rdp_sample_sk(q, alpha, mu_multiplier, gamma, d)
        current_eps = current_eps*rounds +\
            (np.log(1/delta)+(alpha-1)*np.log(1-1/alpha)-np.log(alpha))/(alpha-1)
        if current_eps < best_eps:
            best_eps = current_eps
            best_alpha = alpha

    print('Best eps: ', best_eps, ' is achieved at alpha: ', best_alpha,
        ' with delta fixed to ', delta)