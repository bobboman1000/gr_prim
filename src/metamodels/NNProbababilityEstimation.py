import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors.base import SupervisedFloatMixin
from sklearn.tree import DecisionTreeClassifier

"""
Tuning parameters:  
    weights = 'uniform' vs weights = 'distance'
    regression vs. classification
    nn-algorithm: High-dimensional -> if in doubt: Ball space
                    """

TREES: str = "KNeighborsRegressor"
KNN: str = "KNeighborsClassifier"


@DeprecationWarning
class bNNProbabilityEstimator:

    def __init__(self, method: str, bag_size: int = 1, weights="uniform", algorithm="ball_tree",
                 leaf_size=30, p=2, metric="minkowski", metric_params=None, n_jobs=None):
        self.bag = None
        if method == TREES:
            self.bag_list = self.__compose_bag(bag_size, DecisionTreeClassifier)
        elif method == KNN:
            self.bag_list = self.__compose_bag(bag_size, KNeighborsClassifier, weights=weights,
                                               algorithm=algorithm,
                                               leaf_size=leaf_size, p=p, metric=metric,
                                               metric_params=metric_params,
                                               n_jobs=n_jobs)

    def __compose_bag(self, bag_size: int, construct_function, **func_args):
        bag_list = []
        for i in range(bag_size):
            bag_list.append(construct_function(**func_args))
        return bag_list

    def fit(self, X: pd.DataFrame, y):
        bag = []
        for item in self.bag_list:
            bag.append(self.fit_bag_from_sample(X, y, item))
        return NNProbabilityEstimatorBag(bag)

    def fit_bag_from_sample(self, X: pd.DataFrame, y, item: SupervisedFloatMixin):
        sample_size = X.shape[0]
        bootstrap_sample = X.sample(sample_size, replace=True)
        return item.fit(X=bootstrap_sample, y=y)


class NNProbabilityEstimatorBag:
    def __init__(self, bag: list):
        self.bag: list = bag

    def predict(self, X: pd.DataFrame):
        max_prediction = map(lambda row: np.argmax(row), self.predict_proba(X))
        return np.ndarray(max_prediction)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        count = 0
        assert len(self.bag) > 0
        y: np.ndarray = self.bag[0].predict_proba(X)  # Dummy predict to obtain array size
        for item in self.bag:
            new_y: np.ndarray = item.predict_proba(X)
            y = self.fold_left_columnwise_mean(y, new_y, count, 1)
            count = count + 1

        if y.shape[1] == 1:
            y = self.add_dummy_col(y)
        return y

    """
    Add the second column with 0s if only one class occurred while training. This is bad, but necessary.
    """
    def add_dummy_col(self, y: np.ndarray):
        assert y[:,0].all()  # Only zeros
        print(y.shape[0])
        nd_y = np.ndarray([y.shape[0], 2])
        nd_y[:, 0] = y[:,0]
        nd_y[:, 1] = np.zeros(y.shape[0])
        return nd_y

    def fold_left_columnwise_mean(self, A: np.ndarray, B: np.ndarray, weight_multiplier_A, weight_multiplier_B):
        assert A.shape == B.shape
        A = A * weight_multiplier_A
        B = B * weight_multiplier_B
        sum_weights = weight_multiplier_A + weight_multiplier_B
        A = A + B
        return np.divide(A, sum_weights)
