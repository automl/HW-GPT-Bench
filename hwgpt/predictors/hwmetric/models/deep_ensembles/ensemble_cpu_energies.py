import xgboost as xgb

import json
import logging
import os
import sys
import numpy as np

from sklearn.metrics import mean_squared_error
from typing import List
import xgboost as xgb
from lightgbm import LGBMRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.model_selection import train_test_split
import uncertainty_toolbox as uct
import pickle
def get_ensemble_members(model_type="ensemble_xgb"):
    ensemble_list = []
    if model_type == "ensemble_xgb":
        for depth in [5, 9, 3]:
            for n_estimators in [200, 500, 800]:
                for learning_rate in [0.01, 0.1, 0.1,1.0]:
                    ensemble_list.append(
                        xgb.XGBRegressor(
                            max_depth=depth,
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                            eval_metric="rmse",
                        )
                    )
    elif model_type == "ensemble_lightgbm":
        for depth in [5,  9, 3]:
            for n_estimators in [200, 500,  800]:
                for learning_rate in [0.01,  0.1, 0.001, 1.0]:
                    ensemble_list.append(
                        LGBMRegressor(
                            max_depth=depth,
                            n_estimators=n_estimators,
                            learning_rate=learning_rate,
                        )
                    )
    elif model_type == "ensemble_mix":
        # define all the models
        ensemble_list = [
            xgb.XGBRegressor(),
            xgb.XGBRegressor(n_estimators=500, max_depth=5, learning_rate=0.01),
            xgb.XGBRegressor(n_estimators=800, max_depth=3, learning_rate=0.1),
            xgb.XGBRegressor(n_estimators=200, max_depth=9, learning_rate=1),
            LGBMRegressor(),
            LGBMRegressor(n_estimators=500, max_depth=5, learning_rate=0.01),
            LGBMRegressor(n_estimators=800, max_depth=3, learning_rate=0.1),
            LGBMRegressor(n_estimators=200, max_depth=9, learning_rate=1),
            LinearRegression(),
            Ridge(),
            RandomForestRegressor(n_estimators=400, max_depth=5),
        ]
    
    # else:
    #    raise ValueError("Member not implemented")
    print(ensemble_list)
    return ensemble_list

ensemble_lengths = {
    "ensemble_xgb": 27,
    "ensemble_lightgbm": 27,
    "ensemble_mix": 11
}
class BaggingEnsemble:
    def __init__(self, member_model_type):
        self.ensemble_mean = get_ensemble_members(member_model_type)
        self.ensemble_std = get_ensemble_members(member_model_type)

    def save(self):
        raise NotImplementedError

    def load(self, model_paths):
        raise NotImplementedError

    def train(self, X: np.array, y1: np.array, y2: np.array):
        X = np.array(X)
        y1 = np.array(y1)
        y2 = np.log(np.array(y2))
        for i, regressor in enumerate(self.ensemble_mean):
            regressor.fit(X, y1)
        for i, regressor in enumerate(self.ensemble_std):
            regressor.fit(X, y2)
        

    def validate(self, X: np.array, y_mean: np.array, y_std: np.array):
        X = np.array(X)
        y_mean = np.array(y_mean)
        y_std = np.array(y_std)
        predictions = []
        for i, regressor in enumerate(self.ensemble_mean):
            print(regressor.predict(X).shape)
            predictions.append(regressor.predict(X))
        all_predictions = np.array(predictions).T
        mean= np.mean(all_predictions, axis=-1)
        predictions = []
        for i, regressor in enumerate(self.ensemble_std):
            predictions.append(regressor.predict(X))
        all_predictions = np.array(predictions).T
        std = np.mean(all_predictions, axis=-1)
        std = np.exp(std)
        print("RMSE: ", mean_squared_error(y_mean, mean) ** 0.5)
        print("RMSE: ", mean_squared_error(y_std, std) ** 0.5)
        callibration_scores = uct.metrics.get_all_metrics(mean, std, y_mean)
        print("Callibration scores: ", callibration_scores)
        return mean, std
    

    def predict(self, X: np.array, y_mean: np.array):
        X = np.array(X)
        y_mean = np.array(y_mean)
        predictions = []
        for i, regressor in enumerate(self.ensemble_mean):
            predictions.append(regressor.predict(X))
        all_predictions = np.array(predictions).T
        mean = np.mean(all_predictions, axis=-1)
        predictions = []
        for i, regressor in enumerate(self.ensemble_std):
            predictions.append(regressor.predict(X))
        all_predictions = np.array(predictions).T
        std = np.mean(all_predictions, axis=-1)
        std = np.exp(std)
        callibration_scores = uct.metrics.get_all_metrics(mean, std, y_mean)
        return mean, std, callibration_scores

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--search_space", type=str, default="s")
    parser.add_argument("--device", type=str, default="a6000")
    parser.add_argument("--type", type=str, default="ensemble_lightgbm")
    parser.add_argument("--metric", type=str, default="energies")
    args = parser.parse_args()
    ensemble = BaggingEnsemble(args.type)
    import pandas as pd 
    if args.metric == "energies":
        df = pd.read_csv("gpt_"+args.search_space+"_energies_"+args.device+".csv")
    else:
        df = pd.read_csv("gpt_"+args.search_space+"_latencies_"+args.device+".csv")
    

    target_avg = "Target_Avg"
    target_std = "Target_Std"

    train_df, test_df = train_test_split(df, test_size=0.2, random_state=42, shuffle=True)
    #if args.metric == "energies":
    #    target_cols = [x for x in df.columns if x.startswith("energy")]
    #else:
    #    target_cols = [x for x in df.columns if x.startswith("latency")]
    #target_cols = [x for x in df.columns if x.startswith("energy")]
    # features = [x for x in df.columns if x not in target_cols]

    # train_df[features] = train_df[features].astype("category")
    target_cols = [x for x in df.columns if x.startswith("energy")]
    train_df[target_avg] = train_df["energy_mean"]*1000
    train_df[target_std] = train_df["energy_std"]*1000
    #for target in [target_avg, target_std]:
    #    train_df[target] = np.log(train_df[target] + 1)
    train_df = train_df.drop(columns=target_cols)

    ensemble.train(train_df.drop(columns=[target_avg, target_std]), train_df[target_avg], train_df[target_std])
    test_df[target_avg] = test_df["energy_mean"]*1000
    test_df[target_std] = test_df["energy_std"]*1000
    #for target in [target_avg, target_std]:
    #    test_df[target] = np.log(test_df[target] + 1)
    test_df = test_df.drop(columns=target_cols)
    ensemble.validate(test_df.drop(columns=[target_avg, target_std]), test_df[target_avg], test_df[target_std])
    mean,std, callibration_scores = ensemble.predict(test_df.drop(columns=[target_avg, target_std]), test_df[target_avg])
    # save the model
    save_dir = "ensemble_models/"
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    save_path = save_dir+"ensemble_"+args.type+"_"+args.search_space+"_"+args.device+"_"+args.metric+".pkl"
    with open(save_path, "wb") as f:
        pickle.dump(ensemble, f)
    with open(save_dir+"ensemble_"+args.type+"_"+args.search_space+"_"+args.device+"_"+args.metric+"_callibration.pkl", "wb") as f:
        pickle.dump([test_df,mean,std,callibration_scores], f)
        