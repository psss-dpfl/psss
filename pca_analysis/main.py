import argparse
import logging
import os
import pickle
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from dataset import get_datase
from util import guassian_noise, prob_round_arry, skellam_noise


def setup_log(name: str) -> logging.Logger:
    """Generate the logger for the current run.
    Args:
        name (str): Logging file name.

    Returns:
        logging.Logger: Logger object for the current run.
    """
    my_logger = logging.getLogger(name)
    my_logger.setLevel(logging.INFO)
    log_format = logging.Formatter("%(asctime)s %(levelname)-8s %(message)s")
    filename = f"log_{name}.log"
    log_handler = logging.FileHandler(f"log/{filename}", mode="w")
    log_handler.setLevel(logging.INFO)
    log_handler.setFormatter(log_format)
    my_logger.addHandler(log_handler)
    return my_logger


def PCA(X, num_components):

    # Step-1
    X_meaned = X - np.mean(X, axis=0)

    # Step-2
    cov_mat = np.cov(X_meaned, rowvar=False)

    # Step-3
    eigen_values, eigen_vectors = np.linalg.eigh(cov_mat)

    # Step-4
    sorted_index = np.argsort(eigen_values)[::-1]
    sorted_eigenvectors = eigen_vectors[:, sorted_index]

    # Step-5
    eigenvector_subset = sorted_eigenvectors[:, 0:num_components]

    # Step-6
    X_reduced = np.dot(eigenvector_subset.transpose(), X_meaned.transpose()).transpose()

    return X_reduced, eigenvector_subset


if __name__ == "__main__":
    """Run the main function
    Example run:
    python main.py --dataset iris --clipping_norm 1 --b 6 --random_seed 1234 --setting fl --skellam_mu 1.0 --num_components 2
    python main.py --dataset iris --clipping_norm 1 --b 6 --random_seed 1234 --setting centralized --skellam_mu 1.0 --num_components 2
    """
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", help="dataset name", type=str, default="iris")
    parser.add_argument(
        "--clipping_norm", help="clipping norm for each row", type=float, default=2
    )
    parser.add_argument("--b", help="b for computing gamma", type=float, default=6)
    parser.add_argument("--random_seed", help="random seed", type=int, default=1234)
    parser.add_argument(
        "--sigma", help="sigma for the gaussian distribution", type=float, default=10000
    )
    parser.add_argument(
        "--setting",
        help="setting for the run, fl or centralized",
        type=str,
        default="fl",
    )
    parser.add_argument(
        "--skellam_mu", help="mu for skellam_noise", type=float, default=7
    )
    parser.add_argument(
        "--num_components", help="number of components", type=int, default=2
    )
    args = parser.parse_args()

    log_file = "_".join(
        [
            args.setting,
            args.dataset,
            str(args.random_seed),
            str(args.skellam_mu),
            str(args.sigma),
            str(args.b),
            str(args.clipping_norm),
        ]
    )

    np.random.seed(args.random_seed)
    random.seed(args.random_seed)

    logger = setup_log(log_file)

    IS_PLOT = False
    IS_ANALYSIS = False
    log_dir = "results"
    logger.info(args)

    baseline_time = time.time()
    # 1. Preprocess the data: include the normalization over column and clipping over rows
    X, y = get_datase(args.dataset, args.clipping_norm)
    logger.info("Load data time: %s", time.time() - baseline_time)
    baseline_time = time.time()
    file_name = "_".join(
        [
            args.setting,
            args.dataset,
            str(X.shape[0]),
            str(args.random_seed),
            str(args.skellam_mu),
            str(args.sigma),
            str(args.b),
            str(args.clipping_norm),
        ]
    )

    if os.path.exists(f"{log_dir}/{file_name}.pkl") is False:
        if args.setting == "fl":
            # 2. Discretize the data
            gamma = 2**args.b
            prepare_x = X * gamma
            assert X[10, 2] * gamma == prepare_x[10, 2], "gamma multiplication error"
            discretized_x = prob_round_arry(prepare_x)  # prob_round(prepare_x)
            logger.info("Discretize data time: %s", time.time() - baseline_time)

            # 3. add noise
            # trunk-ignore(git-diff-check/error)
            baseline_time = time.time()
            multiplied_x = np.dot(discretized_x.T, discretized_x)
            assert multiplied_x.shape == (
                X.shape[1],
                X.shape[1],
            ), "multiplication error"
            tri_upper_diag_x = np.triu(multiplied_x, k=0)
            noise_size = tri_upper_diag_x.shape
            mu = args.skellam_mu * (gamma**2 + X.shape[1])
            noise = skellam_noise(noise_size, mu)

            tri_upper_diag_x = np.triu(tri_upper_diag_x + noise, k=0)
            noisy_cov = (
                tri_upper_diag_x
                + tri_upper_diag_x.T
                - np.diag(np.diag(tri_upper_diag_x))
            )
            noisy_cov = noisy_cov / (gamma**2)
            logger.info(
                "l2 distance between the noisy x and true x : %.5f",
                np.linalg.norm(noisy_cov - np.dot(X.T, X)),
            )
            logger.info("Add noise time: %s", time.time() - baseline_time)

        elif args.setting == "centralized":
            multiplied_x = np.dot(X.T, X)
            tri_upper_diag_x = np.triu(multiplied_x, k=0)
            noise_size = tri_upper_diag_x.shape
            noise = guassian_noise(noise_size, args.sigma)
            tri_upper_diag_x = np.triu(tri_upper_diag_x + noise, k=0)
            noisy_cov = (
                tri_upper_diag_x
                + tri_upper_diag_x.T
                - np.diag(np.diag(tri_upper_diag_x))
            )
            logger.info("Add noise time: %s", time.time() - baseline_time)

        elif args.setting == "local":
            noise_size = X.shape
            noise = guassian_noise(noise_size, args.sigma)
            noisy_x = X + noise
            noisy_cov = np.dot(noisy_x.T, noisy_x)

        # 4. SVD for noisy_cov
        baseline_time = time.time()
        eigen_values, eigen_vectors = np.linalg.eigh(noisy_cov)
        sorted_index = np.argsort(eigen_values)[::-1]
        sorted_eigenvalue = eigen_values[sorted_index]
        sorted_eigenvectors = eigen_vectors[:, sorted_index]

        with open(f"{log_dir}/{file_name}.pkl", "wb") as f:
            pickle.dump(
                {"eigenvalue": sorted_eigenvalue, "eigenvector": sorted_eigenvectors}, f
            )
    
    else:
        if IS_ANALYSIS:
            with open(f"{log_dir}/{file_name}.pkl", "rb") as f:
                saved_results = pickle.load(f)
                # trunk-ignore(git-diff-check/error)
                sorted_eigenvalue = saved_results["eigenvalue"]
                sorted_eigenvectors = saved_results["eigenvector"]

    if IS_ANALYSIS:
        for num_components in [2,5,8]:#[10,25,50,100,125,250,500,1000]:
            noisy_eigenvector_subset = sorted_eigenvectors[:, 0 : num_components]
            logger.info("SVD time: %s", time.time() - baseline_time)

            # 5. Measure the performance
            baseline_time = time.time()
            _, clean_eigenvector_subset = PCA(X, num_components=num_components)
            A = np.dot(X.T, X) / X.shape[0]
            q_f = np.trace(
                np.matmul(
                    np.matmul(noisy_eigenvector_subset.transpose(), A), noisy_eigenvector_subset
                )
            )
            q_f_clean = np.trace(
                np.matmul(
                    np.matmul(clean_eigenvector_subset.transpose(), A), clean_eigenvector_subset
                )
            )
            q_a = abs(np.inner(noisy_eigenvector_subset[:, 0], clean_eigenvector_subset[:, 0]))
            logger.info("q_f:  %.8f", q_f)
            logger.info("q_clean: %.8f", q_f_clean)
            logger.info("q_a:  %.8f", q_a)

            results = {
                "setting": [args.setting],
                "dataset": [args.dataset],
                "num_samples": [X.shape[0]],
                "num_components": [num_components],
                "random_seed": [args.random_seed],
                "skellam_mu": [args.skellam_mu],
                "sigma": [args.sigma],
                "b": [args.b],
                "clipping_norm": [args.clipping_norm],
                "q_f": [q_f],
                "q_clean": [q_f_clean],
                "q_a": [q_a],
            }
            df = pd.DataFrame(results)
            # append data frame to CSV file
            df.to_csv("pca_results.csv", mode="a", index=False, header=False)
            logger.info("Measure time: %s", time.time() - baseline_time)

        if IS_PLOT:
            # 6. Show the result and compare with the baseline
            X_reduced = np.dot(
                noisy_eigenvector_subset.transpose(), X.transpose()
            ).transpose()
            principal_df = pd.DataFrame(X_reduced, columns=["PC1", "PC2"])
            principal_df = pd.concat([principal_df, pd.DataFrame(y)], axis=1)
            plt.figure(figsize=(6, 6))
            sns.scatterplot(
                data=principal_df, x="PC1", y="PC2", hue="target", s=60, palette="icefire"
            )

            # 8. also show the result wihtout noise
            clean_principal_df = pd.DataFrame(
                PCA(X, args.num_components)[0], columns=["PC1", "PC2"]
            )
            clean_principal_df = pd.concat([clean_principal_df, pd.DataFrame(y)], axis=1)
            sns.scatterplot(
                data=clean_principal_df, x="PC1", y="PC2", hue="target", s=60, marker="+"
            )
            plt.show()
