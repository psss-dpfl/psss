import numpy as np

from log_maths import _log_add
from log_maths import _log_comb

import argparse

def rdp_sk(alpha, f1_l2, f1_l1, f2_l2, f2_l1, f3_l2, f3_l1, f1_mu, f2_mu, f3_mu):
    sum = beta* (alpha/4/f1_mu * f1_l2*f1_l2 + alpha/4/f2_mu * f2_l2*f2_l2 + alpha/4/f3_mu * f3_l2*f3_l2)
    sum = sum + min(beta*beta*((2*alpha)*f1_l2*f1_l2+6*f1_l1)/16/f1_mu/f1_mu, beta*3*f1_l1/4/f1_mu)
    sum = sum + min(beta*beta*((2*alpha)*f2_l2*f2_l2+6*f2_l1)/16/f2_mu/f2_mu, beta*3*f2_l1/4/f2_mu)
    sum = sum + min(beta*beta*((2*alpha)*f3_l2*f3_l2+6*f3_l1)/16/f3_mu/f3_mu, beta*3*f3_l1/4/f3_mu)
    return sum

def rdp_sample_sk(q, alpha, f1_l2, f1_l1, f2_l2, f2_l1, f3_l2, f3_l1, f1_mu, f2_mu, f3_mu):
    current_log_sum = (alpha-1)*np.log(1-q) + np.log(alpha*q-q+1)
    for k in range(2,alpha+1):
        current_term =  _log_comb(alpha, k)+ (alpha-k)*np.log(1-q) + k*np.log(q) + \
            (k-1)*rdp_sk(alpha, f1_l2, f1_l1, f2_l2, f2_l1, f3_l2, f3_l1, f1_mu, f2_mu, f3_mu)
        current_log_sum = _log_add(current_log_sum, current_term)
    return current_log_sum / (alpha-1)


"""
Example:
  python psss_lr_analysis_client.py --epochs=30 --gamma=100 --d=51 --mu_multipler=100 --beta=1.1
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # DATA
    parser.add_argument("--d", help="dimension of x", type=float)


    # TRAIN
    parser.add_argument("--epochs", help="", type=int)
    parser.add_argument("--gamma", help="scale parameter", type=float, default=1)


    # NOISE
    parser.add_argument("--mu_multipler", help="scale of overall skellam multiplier", type=float, default=0.01)

    # privacy parameter
    parser.add_argument("--delta", help="delta in approximate DP", type=float, default=0.00001)
    parser.add_argument("--beta", help="beta number of clients / number of clients - 1", type=float, default=1)

    args = parser.parse_args()

    print(args)

    d = args.d
    epochs = args.epochs
    gamma = args.gamma
    mu_multipler = args.mu_multipler
    delta = args.delta
    beta = args.beta
    
    rounds = epochs
    
    f1_l2 = np.sqrt(gamma**2 + d)
    f1_l1 = min(np.sqrt(d)*f1_l2, f1_l2*f1_l2)
    
    f2_l2 = np.power(np.sqrt(gamma**2 + d),3)
    f2_l1 = min(np.sqrt(d)*f2_l2, f2_l2*f2_l2)
    
    f3_l2 = gamma * np.sqrt(gamma**2 + d)
    f3_l1 = min(np.sqrt(d)*f3_l2, f3_l2*f3_l2)
    
    mu_1 = mu_multipler * f1_l2 * f1_l2
    mu_2 = mu_multipler * f2_l2 * f2_l2
    mu_3 = mu_multipler * f3_l2 * f3_l2

    best_alpha = 0
    best_eps = np.inf
    alpha_candidates = np.array(range(2,200))
    for alpha in alpha_candidates:
        #print(alpha)
        current_eps = rdp_sk(alpha, 2*f1_l2, 2*f1_l1, 2*f2_l2, 2*f2_l1, 2*f3_l2, 2*f3_l1, mu_1, mu_2, mu_3) 
        current_eps = current_eps*rounds +\
            (np.log(1/delta)+(alpha-1)*np.log(1-1/alpha)-np.log(alpha))/(alpha-1)
        if current_eps < best_eps:
            best_eps = current_eps
            best_alpha = alpha

    print('Best eps: ', best_eps, ' is achieved at alpha: ', best_alpha,
        ' with delta fixed to ', delta)