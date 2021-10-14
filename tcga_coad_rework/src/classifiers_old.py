import numpy as np
from matplotlib.cm import get_cmap
from sklearn.linear_model import LogisticRegressionCV
from sklearn.ensemble import RandomForestClassifier

from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import classification_report
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


def get_n_colors(n):
    """return list of colors from viridis colorspace for use with plotly"""

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
    def __init__(self, df, n_inner_folds=12, n_outer_folds=10, v=0):

        self.df = df
        self.n_outer_folds = n_outer_folds

        self.X = df.drop('tumor_loc_left', axis=1)
        self.y = df['tumor_loc_left']

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
            verbose=v,
            refit=True,
            intercept_scaling=1.0,
            random_state=3257
        )
        self.cv = StratifiedKFold(
            n_splits=self.n_outer_folds,
            random_state=5235)

    def fit_and_print_roc(self):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        plotly_data = []

        colors = get_n_colors(self.n_outer_folds)

        i = 0
        for train, test in self.cv.split(self.X, self.y):
            probabilities_ = self.classifier.fit(
                self.X.iloc[train],
                self.y.iloc[train]
            ).predict_proba(self.X.iloc[test])

            # Compute ROC curve and area under the curve
            fpr, tpr, thresholds = roc_curve(
                self.y.iloc[test],
                probabilities_[:, 1])
            tprs.append(interp(mean_fpr, fpr, tpr))
            tprs[-1][0] = 0.0
            roc_auc = auc(fpr, tpr)
            aucs.append(roc_auc)
            plotly_data.append(
                go.Scatter(
                    x=fpr,
                    y=tpr,
                    name='ROC fold {} (AUC = {})'.format(
                        i, round(roc_auc, 2)),
                    line=dict(color=colors[i], width=1)
                )
            )
            i += 1

        # add ROC reference line
        plotly_data.append(
            go.Scatter(
                x=[0, 1],
                y=[0, 1],
                mode='lines',
                line=dict(
                    color='navy',
                    width=2,
                    dash='dash'
                ),
                showlegend=False
            )
        )

        # mean
        mean_tpr = np.mean(tprs, axis=0)
        mean_tpr[-1] = 1.0
        mean_auc = auc(mean_fpr, mean_tpr)
        std_auc = np.std(aucs)

        # Standard Deviation
        std_tpr = np.std(tprs, axis=0)
        tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
        tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
        plotly_data.append(
            go.Scatter(
                x=mean_fpr,
                y=tprs_upper,
                name='upper bound',
                line=dict(color='grey'),
                opacity=0.1,
                showlegend=False
            )
        )
        # plot mean above std deviation
        plotly_data.append(
            go.Scatter(
                x=mean_fpr,
                y=tprs_lower,
                name='± 1 std. dev.',
                fill='tonexty',
                line=dict(color='grey'),
                opacity=0.1
            )
        )

        plotly_data.append(
            go.Scatter(
                x=mean_fpr,
                y=mean_tpr,
                name='Mean ROC (AUC = {} ± {})'.format(
                    round(mean_auc, 2),
                    round(std_auc, 2)
                ),
                line=dict(color='darkorange', width=3),
            )
        )

        layout = go.Layout(title='Receiver operating characteristic',
                           xaxis=dict(title='False Positive Rate'),
                           yaxis=dict(title='True Positive Rate')
                           )
        fig = go.Figure(data=plotly_data, layout=layout)

        iplot(fig)

    def get_n_pos_most_important(self, r_min, r_max, n=10, plot=True):
        """Return Variants whose Regression Coefficients are largest.

        assumes that the target class is `left`.
        """
        coefs = self.classifier.fit(self.X, self.y).coef_[0]
        indices = np.argsort(coefs)[::-1]

        n_indices = []
        for i in range(n):
            n_indices.append(indices[i])

        if plot:
            trace1 = go.Bar(x=self.df.columns[n_indices],
                            y=coefs[n_indices],
                            text=self.df.columns[n_indices],
                            marker=dict(color='green'),
                            opacity=0.5
                            )

            layout = go.Layout(
                title="Positive Regression Coefficients",
                yaxis=dict(range=[r_min, r_max])
            )
            fig = go.Figure(data=[trace1], layout=layout)

            iplot(fig)

    def get_n_neg_most_important(self, r_min, r_max, n=10, plot=True):
        """Return Variants whose Regression Coefficients are largest.

        assumes that the target class is `left`.
        """
        coefs = self.classifier.fit(self.X, self.y).coef_[0]
        indices = np.argsort(coefs)

        n_indices = []
        for i in range(n):
            n_indices.append(indices[i])

        if plot:
            trace1 = go.Bar(x=self.df.columns[n_indices],
                            y=coefs[n_indices],
                            text=self.df.columns[n_indices],
                            marker=dict(color='green'),
                            opacity=0.5
                            )

            layout = go.Layout(
                title="Negative Regression Coefficients",
                yaxis=dict(range=[r_min, r_max])
            )
            fig = go.Figure(data=[trace1], layout=layout)

            iplot(fig)

    def get_n_most_sided(self, n=10):
        """return two dicts, right and left, each with n case UUIDs
        and respective P(right-sidedness) - P(left-sidedness), where
        this value is minimal in the classifier.

        the classifier must be fit() before using this function.
        """

        # The order of the classes in probas corresponds to that
        # in the attribute classes_, in this case [False, True]

        probas = self.classifier.predict_proba(self.X)
        proba_diffs = np.array(probas[:, 0] - probas[:, 1])

        indices = np.argsort(proba_diffs)

        left_indices = []
        right_indices = []
        for i in range(n):
            left_indices.append(indices[i])
            right_indices.append(np.flip(indices, 0)[i])

        right = {}
        for i in right_indices:
            case_id = self.df.index[i]
            proba_diff = round(proba_diffs[i], 4)
            right[case_id] = proba_diff

        left = {}
        for i in left_indices:
            case_id = self.df.index[i]
            proba_diff = round(proba_diffs[i], 4)
            left[case_id] = proba_diff

        return right, left


class ARandomForestClassifier:
    def __init__(self, df, n_trees=1000, v=0):
        self.df = df
        self.X = df.drop('tumor_loc_left', axis=1)
        self.y = df['tumor_loc_left']
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
            random_state=3576,
            verbose=v,
            warm_start=False,
            class_weight=None
        )

    def fit(self):
        self.classifier.fit(self.X, self.y)

    def print_roc(self):
        tprs = []
        aucs = []
        mean_fpr = np.linspace(0, 1, 100)

        y_decision = self.classifier.oob_decision_function_[:, 1]
        fpr, tpr, thresholds = roc_curve(self.y, y_decision)
        roc_auc = auc(fpr, tpr)

        trace0 = go.Scatter(
            x=fpr,
            y=tpr,
            name='ROC AUC = {}'.format(round(roc_auc, 2)),
            line=dict(color='darkorange', width=2)
        )
        trace1 = go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode='lines',
            line=dict(color='navy', width=2, dash='dash'),
            showlegend=False
        )
        layout = go.Layout(
            title='Receiver operating characteristic',
            xaxis=dict(title='False Positive Rate'),
            yaxis=dict(title='True Positive Rate')
        )

        data = [trace0, trace1]
        fig = go.Figure(data=data, layout=layout)

        iplot(fig)

    def get_n_most_important(self, n=10, plot=True):
        """Return Variants IDs deemed most important by the Classifier.

        Metric used by rf clf is (by default) `gini impurity`, or
        if specified, entropy gain.
        """

        def std_dev_of_features(n=8000):
            """use chunking to get std dev in memory-efficient way

            For large numbers of features and large numbers of trees,
            the memory required to store the input matrix for np.std becomes
            Infeasible. Here, the approach is to scan through all trees
            multiple times, selecting only a batch of features at a time
            and then calculating the std dev on that batch. This way we avoid
            loading the entire matrix of possibilities into memory at once.
            """

            std = np.zeros(self.X.shape[1])
            for i in range(0, self.X.shape[1], n):
                chunk_buffer = []
                for tree in self.classifier.estimators_:
                    chunk_buffer.append(tree.feature_importances_[i:i + n])
                std[i:i + n] = np.std(chunk_buffer, axis=0)
                del chunk_buffer

            return std

        std = std_dev_of_features()
        importances = self.classifier.feature_importances_
        indices = np.flip(np.argsort(importances), 0)

        n_indices = []
        for i in range(n):
            n_indices.append(indices[i])

        if plot:
            # plotly
            trace = go.Bar(x=self.df.columns[n_indices],
                           y=importances[n_indices],
                           text=self.df.columns[n_indices],
                           marker=dict(color='green'),
                           error_y=dict(
                               visible=True,
                               arrayminus=std[n_indices]),
                           opacity=0.5
                           )

            layout = go.Layout(title="Feature importance")
            fig = go.Figure(data=[trace], layout=layout)

            iplot(fig)

        return self.df.columns[n_indices].tolist()

    def get_n_most_sided(self, n=10):
        """return two dicts, right and left, each with n case UUIDs
        and respective P(right-sidedness) - P(left-sidedness), where
        this value is minimal in the classifier.

        the classifier must be fit() before using this function.
        """

        # The order of the classes in probas corresponds to that
        # in the attribute classes_, in this case [False, True]

        probas = self.classifier.predict_proba(self.X)
        proba_diffs = np.array(probas[:, 0] - probas[:, 1])

        indices = np.argsort(proba_diffs)

        left_indices = []
        right_indices = []
        for i in range(n):
            left_indices.append(indices[i])
            right_indices.append(np.flip(indices, 0)[i])

        right = {}
        for i in right_indices:
            case_id = self.df.index[i]
            proba_diff = round(proba_diffs[i], 4)
            right[case_id] = proba_diff

        left = {}
        for i in left_indices:
            case_id = self.df.index[i]
            proba_diff = round(proba_diffs[i], 4)
            left[case_id] = proba_diff

        return right, left
