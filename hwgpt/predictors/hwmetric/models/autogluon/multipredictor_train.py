from __future__ import annotations

import os.path

import pandas as pd
from autogluon.common.utils.utils import setup_outputdir
from autogluon.core.utils.loaders import load_pkl
from autogluon.core.utils.savers import save_pkl
from autogluon.tabular import TabularDataset, TabularPredictor
from sklearn.model_selection import train_test_split
import numpy as np
import uncertainty_toolbox as uct


class MultilabelPredictor:
    """Tabular Predictor for predicting multiple columns in table.
    Creates multiple TabularPredictor objects which you can also use individually.
    You can access the TabularPredictor for a particular label via: `multilabel_predictor.get_predictor(label_i)`.

    Parameters
    ----------
    labels : List[str]
        The ith element of this list is the column (i.e. `label`) predicted by the ith TabularPredictor stored in this object.
    path : str, default = None
        Path to directory where models and intermediate outputs should be saved.
        If unspecified, a time-stamped folder called "AutogluonModels/ag-[TIMESTAMP]" will be created in the working directory to store all models.
        Note: To call `fit()` twice and save all results of each fit, you must specify different `path` locations or don't specify `path` at all.
        Otherwise files from first `fit()` will be overwritten by second `fit()`.
        Caution: when predicting many labels, this directory may grow large as it needs to store many TabularPredictors.
    problem_types : List[str], default = None
        The ith element is the `problem_type` for the ith TabularPredictor stored in this object.
    eval_metrics : List[str], default = None
        The ith element is the `eval_metric` for the ith TabularPredictor stored in this object.
    consider_labels_correlation : bool, default = True
        Whether the predictions of multiple labels should account for label correlations or predict each label independently of the others.
        If True, the ordering of `labels` may affect resulting accuracy as each label is predicted conditional on the previous labels appearing earlier in this list (i.e. in an auto-regressive fashion).
        Set to False if during inference you may want to individually use just the ith TabularPredictor without predicting all the other labels.
    kwargs :
        Arguments passed into the initialization of each TabularPredictor.

    """

    multi_predictor_file = "multilabel_predictor.pkl"

    def __init__(
        self,
        labels,
        path=None,
        problem_types=None,
        eval_metrics=None,
        consider_labels_correlation=True,
        **kwargs,
    ):
        if len(labels) < 2:
            raise ValueError(
                "MultilabelPredictor is only intended for predicting MULTIPLE labels (columns), use TabularPredictor for predicting one label (column).",
            )
        if (problem_types is not None) and (len(problem_types) != len(labels)):
            raise ValueError(
                "If provided, `problem_types` must have same length as `labels`"
            )
        if (eval_metrics is not None) and (len(eval_metrics) != len(labels)):
            raise ValueError(
                "If provided, `eval_metrics` must have same length as `labels`"
            )
        self.path = setup_outputdir(path, warn_if_exist=False)
        self.labels = labels
        self.consider_labels_correlation = consider_labels_correlation
        self.predictors = (
            {}
        )  # key = label, value = TabularPredictor or str path to the TabularPredictor for this label
        if eval_metrics is None:
            self.eval_metrics = {}
        else:
            self.eval_metrics = {labels[i]: eval_metrics[i] for i in range(len(labels))}
        problem_type = None
        eval_metric = None
        for i in range(len(labels)):
            label = labels[i]
            path_i = os.path.join(self.path, "Predictor_" + str(label))
            if problem_types is not None:
                problem_type = problem_types[i]
            if eval_metrics is not None:
                eval_metric = eval_metrics[i]
            self.predictors[label] = TabularPredictor(
                label=label,
                problem_type=problem_type,
                eval_metric=eval_metric,
                path=path_i,
                **kwargs,
            )

    def fit(self, train_data, tuning_data=None, **kwargs):
        """Fits a separate TabularPredictor to predict each of the labels.

        Parameters
        ----------
        train_data, tuning_data : str or autogluon.tabular.TabularDataset or pd.DataFrame
            See documentation for `TabularPredictor.fit()`.
        kwargs :
            Arguments passed into the `fit()` call for each TabularPredictor.
        """
        if isinstance(train_data, str):
            train_data = TabularDataset(train_data)
        if tuning_data is not None and isinstance(tuning_data, str):
            tuning_data = TabularDataset(tuning_data)
        train_data_og = train_data.copy()
        tuning_data_og = tuning_data.copy() if tuning_data is not None else None
        save_metrics = len(self.eval_metrics) == 0
        for i in range(len(self.labels)):
            label = self.labels[i]
            predictor = self.get_predictor(label)
            if not self.consider_labels_correlation:
                labels_to_drop = [l for l in self.labels if l != label]
            else:
                labels_to_drop = [
                    self.labels[j] for j in range(i + 1, len(self.labels))
                ]
            train_data = train_data_og.drop(labels_to_drop, axis=1)
            if tuning_data is not None:
                tuning_data = tuning_data_og.drop(labels_to_drop, axis=1)
            print(f"Fitting TabularPredictor for label: {label} ...")
            predictor.fit(train_data=train_data, tuning_data=tuning_data, **kwargs)
            self.predictors[label] = predictor.path
            if save_metrics:
                self.eval_metrics[label] = predictor.eval_metric
        self.save()

    def predict(self, data, exp=False, **kwargs):
        """Returns DataFrame with label columns containing predictions for each label.

        Parameters
        ----------
        data : str or autogluon.tabular.TabularDataset or pd.DataFrame
            Data to make predictions for. If label columns are present in this data, they will be ignored. See documentation for `TabularPredictor.predict()`.
        kwargs :
            Arguments passed into the predict() call for each TabularPredictor.
        """
        return self._predict(data, as_proba=False, exp=exp, **kwargs)

    def predict_proba(self, data, exp=False, **kwargs):
        """Returns dict where each key is a label and the corresponding value is the `predict_proba()` output for just that label.

        Parameters
        ----------
        data : str or autogluon.tabular.TabularDataset or pd.DataFrame
            Data to make predictions for. See documentation for `TabularPredictor.predict()` and `TabularPredictor.predict_proba()`.
        kwargs :
            Arguments passed into the `predict_proba()` call for each TabularPredictor (also passed into a `predict()` call).
        """
        return self._predict(data, as_proba=True, exp=exp, **kwargs)

    def evaluate(self, data, exp=False, **kwargs):
        """Returns dict where each key is a label and the corresponding value is the `evaluate()` output for just that label.

        Parameters
        ----------
        data : str or autogluon.tabular.TabularDataset or pd.DataFrame
            Data to evalate predictions of all labels for, must contain all labels as columns. See documentation for `TabularPredictor.evaluate()`.
        kwargs :
            Arguments passed into the `evaluate()` call for each TabularPredictor (also passed into the `predict()` call).
        """
        data = self._get_data(data)
        eval_dict = {}
        for label in self.labels:
            print(f"Evaluating TabularPredictor for label: {label} ...")
            predictor = self.get_predictor(label)
            if exp and label == "Target_Std":
                eval_dict[label] = np.exp(predictor.evaluate(data, **kwargs))
            else:
                eval_dict[label] = predictor.evaluate(data, **kwargs)
            eval_dict[label] = predictor.evaluate(data, **kwargs)
            if self.consider_labels_correlation:
                if exp and label == "Target_Std":
                    data[label] = np.exp(data[label])
                else:
                    data[label] = predictor.predict(data, **kwargs)
        return eval_dict

    def save(self):
        """Save MultilabelPredictor to disk."""
        for label in self.labels:
            if not isinstance(self.predictors[label], str):
                self.predictors[label] = self.predictors[label].path
        save_pkl.save(
            path=os.path.join(self.path, self.multi_predictor_file), object=self
        )
        print(
            f"MultilabelPredictor saved to disk. Load with: MultilabelPredictor.load('{self.path}')"
        )

    @classmethod
    def load(cls, path):
        """Load MultilabelPredictor from disk `path` previously specified when creating this MultilabelPredictor."""
        path = os.path.expanduser(path)
        return load_pkl.load(path=os.path.join(path, cls.multi_predictor_file))

    def get_predictor(self, label):
        """Returns TabularPredictor which is used to predict this label."""
        predictor = self.predictors[label]
        if isinstance(predictor, str):
            return TabularPredictor.load(
                path="data_collection/gpt_datasets/predictor_ckpts/hwmetric/autogluon/"
                + predictor
            )
        return predictor

    def _get_data(self, data):
        if isinstance(data, str):
            return TabularDataset(data)
        return data.copy()

    def _predict(self, data, as_proba=False, exp=False, **kwargs):
        data = self._get_data(data)
        if as_proba:
            predproba_dict = {}
        for label in self.labels:
            print(f"Predicting with TabularPredictor for label: {label} ...")
            predictor = self.get_predictor(label)
            if as_proba:
                predproba_dict[label] = predictor.predict_proba(
                    data, exp=exp, as_multiclass=True, **kwargs
                )
            if label == "Target_Std" and exp:
                data[label] = np.exp(predictor.predict(data, **kwargs))
            else:
                data[label] = predictor.predict(data, **kwargs)
        if not as_proba:
            return data[self.labels]
        else:
            return predproba_dict


def run(args):
    # Following https://auto.gluon.ai/stable/tutorials/tabular/advanced/tabular-multilabel.html#inference-and-evaluation
    data_path = "gpt_" + args.search_space + "_" + "latencies_" + args.device + ".csv"
    df = pd.read_csv(data_path)

    time_limit = args.time_limit

    target_avg = "Target_Avg"
    target_std = "Target_Std"

    train_df, test_df = train_test_split(
        df, test_size=0.2, random_state=42, shuffle=True
    )
    target_cols = [x for x in df.columns if x.startswith("latency")]
    if "cpu" in args.device and "energies" in args.metric:
        train_df[target_avg] = train_df["energy_mean"] * 1000
    else:
        train_df[target_avg] = (train_df[target_cols] * 1000).mean(axis=1)
    if "cpu" in args.device and "latencies" in args.metric:
        train_df[target_std] = (train_df[target_cols] * 1000).std(axis=1)
    else:
        if "cpu" in args.device:
            train_df[target_std] = np.log(train_df["energy_std"] * 1000)
        else:
            train_df[target_std] = np.log((train_df[target_cols] * 1000).std(axis=1))
        exp = True
    train_df = train_df.drop(columns=target_cols)

    labels = [target_avg, target_std]  # which columns to predict based on the others
    problem_types = [
        "regression",
        "regression",
    ]  # type of each prediction problem (optional)
    eval_metrics = [
        "r2",
        "r2",
    ]  # ["r2", "r2"]  # metrics used to evaluate predictions for each label (optional)
    save_path = (
        "gpt_latencies_" + args.search_space + "_" + args.device + "_log/"
    )  # args.save_path

    multi_predictor = MultilabelPredictor(
        labels=labels,
        problem_types=problem_types,
        eval_metrics=eval_metrics,
        path=save_path,
    )
    # dynamic_stacking=False, num_stack_levels=1, num_bag_folds=8, num_bag_sets=4, presets="best_quality"
    multi_predictor.fit(
        train_df,
        time_limit=time_limit,
        dynamic_stacking=False,
        num_stack_levels=1,
        num_bag_folds=8,
        num_bag_sets=2,
        presets="best_quality",
    )

    # test_df[features] = test_df[features].astype("category")
    if "cpu" in args.device and "energies" in args.metric:
        test_df[target_avg] = test_df["energy_mean"] * 1000
        test_df[target_std] = test_df["energy_std"] * 1000
    else:
        test_df[target_avg] = (test_df[target_cols] * 1000).mean(axis=1)
        test_df[target_std] = (test_df[target_cols] * 1000).std(axis=1)
    # for target in [target_avg, target_std]:
    #    test_df[target] = np.log(test_df[target] + 1)
    # metrics = uct.metrics.get_all_metrics(predictions, predictions_std, y)
    test_df = test_df.drop(columns=target_cols)

    predictions = multi_predictor.predict(test_df, exp=exp)
    metrics = uct.metrics.get_all_metrics(
        np.array(predictions["Target_Avg"]),
        np.array(predictions["Target_Std"]),
        np.array(test_df["Target_Avg"]),
    )
    print(metrics)
    # save the metrics
    with open(save_path + "calibration_metrics.pkl", "wb") as f:
        import pickle

        pickle.dump([test_df, predictions, metrics], f)
    print("Predictions:  \n", predictions)

    evaluations = multi_predictor.evaluate(test_df, exp=exp)
    print(evaluations)
    print("Evaluated using metrics:", multi_predictor.eval_metrics)

    for label in labels:
        predictor_class = multi_predictor.get_predictor(label)
        predictor_class.leaderboard(test_df, display=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run autogluon surrogates")
    parser.add_argument("--search_space", type=str, default="s")
    parser.add_argument("--device", type=str, default="a6000")
    parser.add_argument("--time_limit", type=int, default=60 * 30)
    parser.add_argument("--save_path", type=str, default="./ag_model")
    parser.add_argument("--metric", type=str, default="latencies")

    args = parser.parse_args()
    run(args)
