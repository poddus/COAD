import logging
from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import roc_curve, auc

from src import config


class LogitClassifier:
    def __init__(self, df: pd.DataFrame, name: str):
        self.df = df
        self.X = df.drop('tumor_left', axis=1)
        self.y = df['tumor_left']
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_groups()
        self.classifier = LogisticRegression(random_state=config.CLASSIFIER_RANDOM_STATE)
        self.name = name

    def split_groups(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        return train_test_split(
            self.X, self.y,
            test_size=config.CLASSIFIER_TEST_SIZE, random_state=config.CLASSIFIER_RANDOM_STATE
        )

    def fit_whole(self) -> None:
        self.classifier.fit(self.X, self.y)

    def fit_train(self) -> None:
        self.classifier.fit(self.X_train, self.y_train)

    def eval_roc(self) -> Tuple[List, List, List]:
        y_score = self.classifier.decision_function(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, y_score)
        return fpr, tpr, thresholds

    def print_roc_curve(self, fpr, tpr) -> None:
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = {})'.format(auc(fpr, tpr)))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of Logit Classifier on {} data'.format(self.name))
        plt.legend(loc="lower right")
        plt.show()

    def train_test_roc(self) -> None:
        self.fit_train()
        fpr, tpr, threshold = self.eval_roc()
        self.print_roc_curve(fpr, tpr)

    def get_n_most_important(self, n: int = 10, sign: 'str' = 'positive', plot: bool = True):
        """Return Variants whose Regression Coefficients are largest."""

        coefs = pd.DataFrame(index=self.X.columns, data=self.classifier.fit(self.X, self.y).coef_[0])

        if sign == 'positive':
            n_coefs = coefs.sort_values(by=0).tail(n)[::-1]
        elif sign == 'negative':
            n_coefs = coefs.sort_values(by=0).head(n).abs()
        else:
            raise ValueError('`sign` may only be either `positive` or negative, got {}'.format(sign))

        logging.info('{} most important {} features:\n{}'.format(n, sign, n_coefs))

        if plot:
            fig, ax = plt.subplots()
            y_pos = np.arange(len(n_coefs))

            ax.barh(y_pos, n_coefs[0].values)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(n_coefs.index)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('|Coefficient|')
            ax.set_title('{} most important {} features, Logistic Regression'.format(n, sign))

            plt.show()


class RFClassifier:
    def __init__(self, df: pd.DataFrame, name: str, n_trees: int = 6000, max_depth: int = 13, v: int = 1):
        self.X = df.drop('tumor_left', axis=1)
        self.y = df['tumor_left']
        self.classifier = RandomForestClassifier(
            n_estimators=n_trees,
            criterion='gini',
            max_features='sqrt',
            max_depth=max_depth,
            min_samples_split=2,
            min_samples_leaf=1,
            min_weight_fraction_leaf=0.0,
            max_leaf_nodes=None,
            min_impurity_decrease=0.0,
            bootstrap=True,
            oob_score=True,
            n_jobs=-1,
            random_state=config.CLASSIFIER_RANDOM_STATE,
            verbose=v,
            warm_start=False,
            class_weight=None
        )
        self.name = name

    def fit_whole(self):
        # TODO: generates Warning
        #  UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
        self.classifier.fit(self.X, self.y)

    def eval_roc(self) -> Tuple[List, List, List]:
        y_score = self.classifier.oob_decision_function_[:, 1]
        y_score_df = pd.DataFrame(data=y_score, index=self.X.index)
        fpr, tpr, thresholds = roc_curve(self.y, y_score_df)
        return fpr, tpr, thresholds

    def print_roc_curve(self, fpr, tpr):
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = {})'.format(auc(fpr, tpr)))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('ROC of Random Forest Classifier on {} data'.format(self.name))
        plt.legend(loc="lower right")
        plt.show()

    def train_test_roc(self):
        """convenience function for evaluating a classifier"""
        self.fit_whole()
        fpr, tpr, threshold = self.eval_roc()
        self.print_roc_curve(fpr, tpr)

    def get_n_most_important(self, n: int = 10, plot: bool = True):
        importances = self.classifier.feature_importances_
        std = np.std([tree.feature_importances_ for tree in self.classifier.estimators_], axis=0)

        coefs = pd.DataFrame(index=self.X.columns, data=importances)
        coefs[1] = std
        n_coefs = coefs.sort_values(by=0).tail(n)[::-1]

        logging.info('{} most important features:\n{}'.format(n, n_coefs))

        if plot:
            fig, ax = plt.subplots()
            y_pos = np.arange(len(n_coefs))

            ax.barh(y_pos, n_coefs[0].values, xerr=n_coefs[1].values)
            ax.set_yticks(y_pos)
            ax.set_yticklabels(n_coefs.index)
            ax.invert_yaxis()  # labels read top-to-bottom
            ax.set_xlabel('Gini Importance')
            ax.set_title('{} most important features, Random Forest'.format(n))

            plt.show()


class RFClassifierTuningCV:
    def __init__(self, df: pd.DataFrame, n_folds: int = 6, verbose=1) -> None:
        self.df = df
        self.X = df.drop('tumor_left', axis=1)
        self.y = df['tumor_left']
        self.classifier = RandomForestClassifier(n_jobs=-1)
        self.param_grid = {
            'n_estimators': [4000, 5000, 6000, 7000],
            'max_depth': [9, 11, 13, 15, 17]
        }
        self.model = GridSearchCV(
            estimator=self.classifier,
            param_grid=self.param_grid,
            scoring='accuracy', cv=n_folds,
            verbose=verbose
        )

    def fit(self):
        self.model.fit(self.X, self.y)

    def best_params(self):
        return self.model.best_params_
