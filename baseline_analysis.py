import numpy as np
from scipy.special import erfc


from log_maths import _log_add
from log_maths import _log_comb

import argparse


"""
Privacy analysis for baseline mechanism. 
For LR, c=2; for PCA c=1
Example:
  python baseline_analysis.py --eps=1.28 --delta=0.00001 --task=LR --step_start=0 --step_end=1000 --step_chi=0.001 --prec=0.0001
"""


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    
    # Privacy
    parser.add_argument("--eps", help="target eps", type=float)
    parser.add_argument("--delta", help="target delta", type=float)

    parser.add_argument("--task", help="which task", type=str, choices=["LR", "PCA"])
    
    parser.add_argument("--step_end", type=int)
    parser.add_argument("--step_start", type=int)
    parser.add_argument("--step_chi", type=float)
    parser.add_argument("--prec", help="relative prec", type=float)

    args = parser.parse_args()

    eps = args.eps
    delta = args.delta
    c = 1
    
    found = 0
    min_rel_err = 100
    best_sigma = -1
    best_delta = -1
    best_chi_step = -1
    for step in range(args.step_start, args.step_end):
        chi = step*args.step_chi
        cur_delta = (erfc(chi) - np.exp(eps)*erfc(np.sqrt(chi*chi+eps))) / 2
        rel_err = np.abs((cur_delta-delta) / delta)
        sigma = c / np.sqrt(2) / (np.sqrt(chi*chi+eps)-chi)
        if np.abs(rel_err) <= args.prec:
            print('chi is found:', chi)
            print('rel err is:', rel_err)
            print('sigma is:', sigma)
            print('delta is:', cur_delta)
            found = 1
            break
        if abs(rel_err) < min_rel_err:
            min_rel_err = abs(rel_err)
            best_sigma = sigma
            best_delta = cur_delta
            best_chi_step = step
    
    if found == 0:
        print('chi not found, min rel error:', min_rel_err)
        print('sigma is:', best_sigma)
        print('delta is:', best_delta)
        print('chi_step is:', best_chi_step)
        print('chi is:', best_chi_step*args.step_chi)

        