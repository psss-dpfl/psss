import logging
import os
import pickle
import time

import pandas as pd
from folktables import ACSDataSource, ACSIncome, generate_categories
from util import clipping_array, clipping_features, normalize_array, normalized_df


def get_datase(name: str, clipping_norm: float = 1):
    """Load the data with pandas DataFrame format. The last column is the target.

    Args:
        name (str): Name of the datasets. Currently, only 'iris' is supported.
        clipping_norm (float, optional): Clipping norm for each row. Defaults to 1.
    Returns:
        pd.DataFrame: It contains the x and targets (last column).
    """
    start_time = time.time()
    assert name in ["iris", "ca_income"], "dataset name error"
    if os.path.exists(f"dataset/{name}_{clipping_norm}.pkl") is False:
        if name == "iris":
            url = "https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data"
            data = pd.read_csv(
                url,
                names=[
                    "sepal length",
                    "sepal width",
                    "petal length",
                    "petal width",
                    "target",
                ],
            )
            logging.info("Time to load the data: %s seconds", time.time() - start_time)
            assert not data.isnull().values.any(), "NAN in the dataset"
            X = data.iloc[:, 0:-1].to_numpy()
            y = data.iloc[:, -1].to_numpy()

        elif name == "ca_income":
            data_source = ACSDataSource(
                survey_year="2018", horizon="1-Year", survey="person"
            )
            ca_data = data_source.get_data(states=["CA"], download=True)
            definition_df = data_source.get_definitions(download=True)
            categories = generate_categories(
                features=ACSIncome.features, definition_df=definition_df
            )
            X, y, _ = ACSIncome.df_to_pandas(
                ca_data, categories=categories, dummies=True
            )
            X["label"] = y
            X = X.dropna()
            assert not X.isnull().values.any(), "NAN in the dataset"
            y = X.iloc[:, -1].to_numpy()
            X = X.iloc[:, 0:-1].to_numpy()        
        start_time = time.time()
        # normalized_x = normalized_df(X)
        normalized_x = normalize_array(X, "mean")
        logging.info("Time to normalize the data: %s seconds", time.time() - start_time)
        start_time = time.time()
        clipped_x = clipping_array(normalized_x, clipping_norm)
        logging.info("Time to clip the data: %s seconds", time.time() - start_time)
        with open(f"dataset/{name}_{clipping_norm}.pkl", "wb") as f:
            pickle.dump({"X": clipped_x, "y": y}, f)
        logging.info("Shape of X:", clipped_x.shape[0],clipped_x.shape[1])
        return clipped_x, y

    else:
        with open(f"dataset/{name}_{clipping_norm}.pkl", "rb") as f:
            datasets = pickle.load(f)
            logging.info("Shape of X:", datasets["X"].shape[0],datasets["X"].shape[1])
            return datasets["X"], datasets["y"]
