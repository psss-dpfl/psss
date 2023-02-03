import argparse

import numpy as np


def rdp_sk(alpha, f1_l2, f1_l1, f1_mu):
    sum = alpha/4/f1_mu * f1_l2*f1_l2 
    sum = sum + min(((2*alpha)*f1_l2*f1_l2+6*f1_l1)/16/f1_mu/f1_mu, 3*f1_l1/4/f1_mu)
    return sum


"""
Example:
  python psss_pca_analysis.py --gamma=64 --d=816 --mu_multipler=100
"""

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # DATA
    parser.add_argument("--d", help="dimension of x", type=float)

    # TRAIN
    parser.add_argument("--gamma", help="scale parameter", type=float, default=1)

    # NOISE
    parser.add_argument(
        "--mu_multipler",
        help="scale of overall skellam multiplier",
        type=float,
        default=0.01,
    )

    # privacy meter
    parser.add_argument(
        "--delta", help="delta in approximate DP", type=float, default=0.00001
    )

    args = parser.parse_args()

    print(args)

    d = args.d
    gamma = args.gamma
    mu_multipler = args.mu_multipler
    delta = args.delta
<<<<<<< HEAD

    l2 = np.sqrt(gamma**2 + d)
    l1 = min(np.sqrt(d) * l2, l2 * l2)

    mu = mu_multipler * l2 * l2

    best_alpha = 0
    best_eps = np.inf
    alpha_candidates = np.array(range(2, 500))
=======
        
    f1_l2 = np.sqrt(gamma**2 + d)
    f1_l1 = min(np.sqrt(d)*f1_l2, f1_l2*f1_l2)
    
    mu_1 = mu_multipler * f1_l2 * f1_l2

    best_alpha = 0
    best_eps = np.inf
    alpha_candidates = np.array(range(2,200))
>>>>>>> a9b25c6f50c8605451b03bb1e733ebafb5969059
    for alpha in alpha_candidates:
        # print(alpha)
        current_eps = np.inf
<<<<<<< HEAD
        current_eps = rdp_sk(alpha, l2, l1, mu) + (
            np.log(1 / delta) + (alpha - 1) * np.log(1 - 1 / alpha) - np.log(alpha)
        ) / (alpha - 1)
=======
        current_eps = rdp_sk(alpha, f1_l2, f1_l1, mu_1, ) 
        current_eps = current_eps +\
            (np.log(1/delta)+(alpha-1)*np.log(1-1/alpha)-np.log(alpha))/(alpha-1)
>>>>>>> a9b25c6f50c8605451b03bb1e733ebafb5969059
        if current_eps < best_eps:
            best_eps = current_eps
            best_alpha = alpha

<<<<<<< HEAD
    print(
        "Best eps: ",
        best_eps,
        " is achieved at alpha: ",
        best_alpha,
        " with delta fixed to ",
        delta,
        "\n mu add to each entry of X^TX is",
        mu,
    )
=======
    print('Best eps: ', best_eps, ' is achieved at alpha: ', best_alpha,
        ' with delta fixed to ', delta)
>>>>>>> a9b25c6f50c8605451b03bb1e733ebafb5969059
