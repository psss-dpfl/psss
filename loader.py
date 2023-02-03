import logging
import os
import pickle
import time
import pandas as pd
import numpy as np
import torch
import torch.utils.data as data_utils
from sklearn.model_selection import train_test_split
import tensorflow as tf

def clipping_array(data: np.ndarray, clipping_norm: float = 1):
    """
    Clips the data in the array
    """
    t = tf.constant(data, dtype=tf.float32)
    clipped_data = tf.clip_by_norm(t, clipping_norm, axes=1)
    return clipped_data.numpy()


def normalize_array(data: np.ndarray, type: str):
    if type == "minmax":
        min_value = data.min(axis=0)
        max_value = data.max(axis=0)
        data = (data - min_value) / (max_value - min_value + 1e-8)
    elif type == "mean":
        mean = data.mean(axis=0)
        std = data.std(axis=0)
        data = (data - mean) / (std + 1e-8)

    return data


def load_data(data_type, split_size=0.2, row_clip=1, gaussian_sigma=0):
    baseline_time = time.time()
    normalize_type = "mean"
    file_name = f"processed_dataset/{data_type}_{normalize_type}_{row_clip}_{split_size}_{gaussian_sigma}.pkl"
    if os.path.exists(file_name) is False:
        df_X = pd.read_csv(f"dataset/{data_type.lower()}_features.csv", ",")
        df_Y = pd.read_csv(f"dataset/{data_type.lower()}_labels.csv", ";")
        
        # TODO: check whether need to drop na
        df_X["label"] = df_Y.iloc[:, 0]
        df_X = df_X.dropna()
        assert not df_X.isnull().values.any(), "NAN in the dataset"
        Y = df_X.iloc[:, -1].to_numpy()
        X = df_X.iloc[:, 0:-1].to_numpy()    
        
        # X = df_X.to_numpy()
        # Y = df_Y.iloc[:, 0].to_numpy()
        
        # convert the pandas to array 
        normalized_X = normalize_array(X, normalize_type)
        logging.info(f"Normalizing the data costs: %.5f", time.time()-baseline_time)
        baseline_time = time.time()
        clipped_X = clipping_array(normalized_X, row_clip)
        logging.info(f"Clipping the data costs: %.5f", time.time()-baseline_time)
        baseline_time = time.time()
        # split to train and test
        X_train, X_test, Y_train, Y_test = train_test_split(clipped_X, Y, test_size=split_size,shuffle=True)
        print('shape of X_train', X_train.shape)
        print('shape of X_test', X_test.shape)
        
        # convert to tensor
        X_train_tensor = torch.tensor(X_train).float()
        Y_train_tensor = torch.tensor(Y_train).float()
        X_test_tensor = torch.tensor(X_test).float()
        Y_test_tensor = torch.tensor(Y_test).float()

        if gaussian_sigma > 0:
            X_train_tensor = X_train_tensor + torch.normal(mean=0, std=gaussian_sigma, size=X_train_tensor.size())
            Y_train_tensor = Y_train_tensor + torch.normal(mean=0, std=gaussian_sigma, size=Y_train_tensor.size())

        train_tensor = data_utils.TensorDataset(X_train_tensor, Y_train_tensor)
        test_tensor = data_utils.TensorDataset(X_test_tensor, Y_test_tensor)
        logging.info(f"Covert to tensor costs: %.5f", time.time()-baseline_time)

        #TODO: Save the normalized datasets to disk.
        
        with open(file_name,"wb") as f:
            pickle.dump((train_tensor, test_tensor, X_train.shape[0], X_train.shape[1], 1),f)    
        return train_tensor, test_tensor, X_train.shape[0], X_train.shape[1], 1
    else:
        with open(file_name, "rb") as f:
            train_tensor, test_tensor, n_train,n_test, d = pickle.load(f)
        
        return train_tensor, test_tensor, n_train,n_test, d
    
