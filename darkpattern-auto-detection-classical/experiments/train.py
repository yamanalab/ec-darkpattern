from typing import Sequence, Union

import hydra
import lightgbm as lgb
import numpy as np
import optuna
from omegaconf import DictConfig
from optuna.trial import FrozenTrial, Trial
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score, roc_auc_score)
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC

from const.model import (LIGHTGBM_MODEL, LOGISTIC_REGRESSION_MODEL,
                         RANDOM_FOREST_MODEL, SVC_MODEL)
from const.path import CONFIG_PATH, DATASET_TSV_PATH
from utils import dataset
from utils.random_seed import set_random_seed


def get_sklearn_model(
    trial: Union[Trial, FrozenTrial],
    model_name: str = RANDOM_FOREST_MODEL,
    random_state: int = 42,
) -> BaseEstimator:

    if model_name == RANDOM_FOREST_MODEL:
        params = {
            "max_depth": trial.suggest_int("max_depth", 2, 32),
            "n_estimators": trial.suggest_int("n_estimators", 10, 1000),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 32),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 32),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "random_state": random_state,
        }
        return RandomForestClassifier(**params)

    elif model_name == SVC_MODEL:
        params = {
            "C": trial.suggest_loguniform("C", 1e-8, 10.0),
        }
        return SVC(**params)

    elif model_name == LOGISTIC_REGRESSION_MODEL:
        params = {
            "C": trial.suggest_loguniform("C", 1e-8, 10.0),
        }
        return LogisticRegression(**params)

    else:
        raise ValueError("Unknown model name: {}".format(model_name))


def run_lightgbm(
    study: optuna.Study, n_trial: int = 100, random_state: int = 42
) -> None:

    texts, labels = dataset.get_dataset(DATASET_TSV_PATH)
    count_vectorizer = CountVectorizer(ngram_range=(1, 1), dtype=np.float32)
    X, y = (
        count_vectorizer.fit_transform(texts),
        np.array(labels),
    )

    def objective(trial: Trial) -> Union[float, Sequence[float]]:

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)
        f1_scores = list()

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "random_state": random_state,
        }

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            lgbm_train_dataset = lgb.Dataset(X_train, label=y_train)

            gbm = lgb.train(params, lgbm_train_dataset)

            y_preds = np.rint(gbm.predict(X_test))

            f1_scores.append(
                f1_score(y_test, y_preds),
            )

        return np.mean(f1_scores)

    def detailed_objective(trial: FrozenTrial) -> None:

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        accuracy_scores = list()
        f1_scores = list()
        precision_scores = list()
        recall_scores = list()
        roc_auc_scores = list()

        params = {
            "objective": "binary",
            "metric": "binary_logloss",
            "lambda_l1": trial.suggest_loguniform("lambda_l1", 1e-8, 10.0),
            "lambda_l2": trial.suggest_loguniform("lambda_l2", 1e-8, 10.0),
            "num_leaves": trial.suggest_int("num_leaves", 2, 256),
            "feature_fraction": trial.suggest_uniform("feature_fraction", 0.4, 1.0),
            "bagging_fraction": trial.suggest_uniform("bagging_fraction", 0.4, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 1, 7),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "random_state": random_state,
        }

        for train_index, test_index in skf.split(X, y):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]

            lgbm_train_dataset = lgb.Dataset(X_train, label=y_train)

            gbm = lgb.train(params, lgbm_train_dataset)

            output = gbm.predict(X_test)
            y_preds = np.rint(output)
            y_probs = np.array(output)

            (accuracy, f1, precision, recall, roc_auc) = (
                accuracy_score(y_test, y_preds),
                f1_score(y_test, y_preds),
                precision_score(y_test, y_preds),
                recall_score(y_test, y_preds),
                roc_auc_score(y_test, y_probs),
            )

            accuracy_scores.append(accuracy)
            f1_scores.append(f1)
            precision_scores.append(precision)
            recall_scores.append(recall)
            roc_auc_scores.append(roc_auc)

        print(
            {
                "accuracy_scores": accuracy_scores,
                "f1_scores": f1_scores,
                "precision_scores": precision_scores,
                "recall_scores": recall_scores,
                "accuracy_score_average": np.mean(accuracy_scores),
                "f1_score_average": np.mean(f1_scores),
                "precision_score_average": np.mean(precision_scores),
                "recall_score_average": np.mean(recall_scores),
                "roc_auc_score_average": np.mean(roc_auc_scores),
            }
        )

    study.optimize(objective, n_trials=n_trial)

    detailed_objective(study.best_trial)


def run_sklearn(
    study: optuna.Study, model_name: str, n_trial: int = 100, random_state: int = 42
) -> None:
    def objective(trial: Trial) -> Union[float, Sequence[float]]:

        texts, labels = dataset.get_dataset(DATASET_TSV_PATH)

        count_vectorizer = CountVectorizer(ngram_range=(1, 1))

        X, y = (
            count_vectorizer.fit_transform(texts),
            np.array(labels),
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        clf = get_sklearn_model(trial, model_name)
        f1_score_names = ["f1"]
        scores = cross_validate(clf, X, y, cv=skf, scoring=f1_score_names)
        return scores["test_f1"].mean()

    def detailed_objective(trial: FrozenTrial) -> None:

        texts, labels = dataset.get_dataset(DATASET_TSV_PATH)

        count_vectorizer = CountVectorizer(ngram_range=(1, 1))

        X, y = (
            count_vectorizer.fit_transform(texts),
            np.array(labels),
        )

        skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

        clf = get_sklearn_model(trial, model_name)
        score_names = ["accuracy", "f1", "precision", "recall", "roc_auc"]
        scores = cross_validate(clf, X, y, cv=skf, scoring=score_names)

        print(
            {
                "accuracy_scores": list(scores["test_accuracy"]),
                "f1_scores": list(scores["test_f1"]),
                "precision_scores": list(scores["test_precision"]),
                "recall_scores": list(scores["test_recall"]),
                "accuracy_score_average": np.mean(scores["test_accuracy"]),
                "f1_score_average": np.mean(scores["test_f1"]),
                "precision_score_average": np.mean(scores["test_precision"]),
                "recall_score_average": np.mean(scores["test_recall"]),
                "roc_auc_score_average": np.mean(scores["test_roc_auc"]),
            }
        )

    study.optimize(objective, n_trials=n_trial)

    detailed_objective(study.best_trial)


@hydra.main(config_path=CONFIG_PATH, config_name="config.yaml")
def main(cfg: DictConfig) -> None:

    study = optuna.create_study(direction="maximize")

    model_name = cfg.model.name

    set_random_seed(cfg.random.seed)

    if model_name == LIGHTGBM_MODEL:
        run_lightgbm(study, cfg.experiment.n_trial, cfg.random.seed)
    else:
        run_sklearn(study, model_name, cfg.experiment.n_trial, cfg.random.seed)


def init() -> None:
    import nltk

    try:
        nltk.data.find("wordnet")
    except LookupError:
        nltk.download("wordnet")


if __name__ == "__main__":
    init()
    main()
