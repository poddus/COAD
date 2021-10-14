from typing import List, Tuple
import numpy as np
import pandas as pd
from matplotlib.cm import get_cmap
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegressionCV, LogisticRegression
from sklearn.ensemble import RandomForestClassifier
import seaborn as sns
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

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


class LogitClassifier:
    def __init__(self, df: pd.DataFrame):
        self.X = df.drop('tumor_left', axis=1)
        self.y = df['tumor_left']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=config.CLASSIFIER_TEST_SIZE, random_state=config.CLASSIFIER_RANDOM_STATE)
        self.classifier = LogisticRegression()

    def fit_whole(self):
        self.classifier.fit(self.X, self.y)

    def fit_train(self):
        self.classifier.fit(self.X_train, self.y_train)

    def evaluate_on_test(self) -> Tuple[List, List, List]:
        y_score = self.classifier.decision_function(self.X_test)
        fpr, tpr, thresholds = roc_curve(self.y_test, y_score)
        return fpr, tpr, thresholds

    @staticmethod
    def print_roc_curve(fpr, tpr):
        plt.figure()
        lw = 2
        plt.plot(fpr, tpr, color='darkorange',
                 lw=lw, label='ROC curve (area = {})'.format(auc(fpr, tpr)))
        plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
        plt.xlim([0.0, 1.0])
        plt.ylim([0.0, 1.05])
        plt.xlabel('False Positive Rate')
        plt.ylabel('True Positive Rate')
        plt.title('Receiver operating characteristic example')
        plt.legend(loc="lower right")
        plt.show()

    def evaluate(self):
        """convenience function for evaluating a classifier"""
        self.fit_train()
        fpr, tpr, threshold = self.evaluate_on_test()
        self.print_roc_curve(fpr, tpr)


class RFClassifier:
    def __init__(self, df: pd.DataFrame):
        self.X = df.drop('tumor_left', axis=1)
        self.y = df['tumor_left']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            self.X, self.y,
            test_size=config.CLASSIFIER_TEST_SIZE, random_state=config.CLASSIFIER_RANDOM_STATE)



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
