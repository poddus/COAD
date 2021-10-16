from abc import ABC, abstractmethod
from typing import List, Tuple
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, auc

from src import config


def get_n_colors(n) -> List:
    """return list of colors from viridis colorspace"""

    cmap = get_cmap('viridis')
    colors_01 = cmap(np.linspace(0, 1, n))
    colors_255 = []

    for row in colors_01:
        colors_255.append(
            'rgba({}, {}, {}, {}'.format(
                row[0] * 255,
                row[1] * 255,
                row[2] * 255,
                row[3]
            )
        )

    return colors_255


class Classifier(ABC):
    # @abstractmethod
    # def split_groups(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #     """split data into train and test groups"""
    @abstractmethod
    def fit_whole(self) -> None:
        """fit classifier on entire data set"""

    @abstractmethod
    def fit_train(self) -> None:
        """fit classifier on training data set"""

    @abstractmethod
    def eval_roc(self) -> Tuple[List, List, List]:
        """extract fpr, tpr and thresholds of ROC using test data set"""

    @abstractmethod
    def print_roc_curve(self, fpr, tpr) -> None:
        """generate pyplot roc plot and show"""

    @abstractmethod
    def train_test_roc(self) -> None:
        """convenience function for training and evaluating a classifier"""


class LogitClassifier(Classifier):
    def __init__(self, df: pd.DataFrame, name: str):
        self.X = df.drop('tumor_left', axis=1)
        self.y = df['tumor_left']
        self.X_train, self.X_test, self.y_train, self.y_test = self.split_groups()
        self.classifier = LogisticRegression()
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


class RFClassifier(Classifier):
    def __init__(self, df: pd.DataFrame, name: str, n_trees=1000, v=1):
        self.X = df.drop('tumor_left', axis=1)
        self.y = df['tumor_left']
        self.classifier = RandomForestClassifier(
            n_estimators=n_trees,
            criterion='gini',
            max_features='sqrt',
            max_depth=None,
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

    # def split_groups(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    #     return train_test_split(
    #         self.X, self.y,
    #         test_size=config.CLASSIFIER_TEST_SIZE, random_state=config.CLASSIFIER_RANDOM_STATE
    #     )

    def fit_whole(self):
        # TODO: generates Warning
        #  UserWarning: X does not have valid feature names, but RandomForestClassifier was fitted with feature names
        self.classifier.fit(self.X, self.y)

    def fit_train(self):
        # self.classifier.fit(self.X_train, self.y_train)
        pass

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


class LogitClassifierTuningCV:
    def __init__(self, df: pd.DataFrame, n_inner_folds: int = 12, n_outer_folds: int = 10, verbose=1) -> None:
        self.df = df
        self.n_outer_folds = n_outer_folds

        self.X = df.drop('tumor_left', axis=1)
        self.y = df['tumor_left']

        self.classifier = LogisticRegressionCV(
            Cs=10,
            fit_intercept=True,
            cv=n_inner_folds,
            dual=False,
            penalty='l1',
            solver='liblinear',
            tol=0.0001,
            max_iter=100,
            class_weight=None,
            n_jobs=-1,
            verbose=verbose,
            refit=True,
            intercept_scaling=1.0
        )
        self.outer_cv = StratifiedKFold(n_splits=self.n_outer_folds)

    def fit(self):
        self.classifier.fit(self.X, self.y)

    def generate_roc_with_nested_cv(self):
        pass
