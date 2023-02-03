import numpy as np

import argparse

def rdp_sk(alpha, l2, l1, mu):
    sum = alpha/4/mu * l2*l2 
    sum = sum + min(((2*alpha)*l2*l2+6*l1)/16/mu/mu, 3*l1/4/mu)
    return sum

"""
Example:
  python psss_pca_analysis.py --gamma=100 --d=51 --eps=1 --step_start=0 --step_end=300000 --step_mu_multiplier=0.0001 --prec=0.001
"""

def compute_best_eps(l2, l1, mu, delta):
    best_eps = np.inf
    alpha_candidates = np.array(range(2,500))
    for alpha in alpha_candidates:
        # print(alpha)        
        current_eps = rdp_sk(alpha, l2, l1, mu) 
        # print(current_eps)
        # print(1/delta)
        current_eps = current_eps + (np.log(1/delta)+(alpha-1)*np.log(1-1/alpha)-np.log(alpha))/(alpha-1)
        if current_eps < best_eps:
            best_eps = current_eps
    return best_eps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # DATA
    parser.add_argument("--d", help="dimension of x", type=float)

    # Privacy
    parser.add_argument("--eps", help="target eps", type=float)
    parser.add_argument("--delta", help="target delta", type=float, default=0.00001)
    parser.add_argument("--step_end", type=int)
    parser.add_argument("--step_start", type=int)
    parser.add_argument("--step_mu_multiplier", type=float)
    parser.add_argument("--prec", help="relative prec", type=float)

    parser.add_argument("--gamma", help="scale parameter", type=float, default=1)

    args = parser.parse_args()

    print(args)

    d = args.d
    gamma = args.gamma
    delta = args.delta
    eps = args.eps
   
    l2 = np.sqrt(gamma**2 + d)
    l1 = min(np.sqrt(d)*l2, l2*l2)

    
    found = 0
    min_rel_err = 100
    best_mu = -1
    best_eps = -1
    best_mu_multi_step = -1
    for step in range(args.step_start, args.step_end):
        mu_multiplier = step*args.step_mu_multiplier
        mu = mu_multiplier * l2 * l2
        # print('mu', mu)

        cur_eps = compute_best_eps(l2, l1, mu, delta)
        rel_err = np.abs(cur_eps-eps) / eps
        if np.abs(rel_err) <= args.prec:
            print('mu is found:', mu)
            print('rel err is:', rel_err)
            print('eps is:', cur_eps)
            found = 1
            break
        if abs(rel_err) < min_rel_err:
            min_rel_err = abs(rel_err)
            best_mu = mu
            best_eps = cur_eps
            best_mu_multi_step = step
    
    if found == 0:
        print('mu not found, min rel error:', min_rel_err)
        print('best mu is:', best_mu)
        print('best eps is:', best_eps)
        print('best mu_step is:', best_mu_multi_step)