import logging
import traceback
from typing import List, Union

import numpy as np
from sklearn import mixture
from sklearn.mixture import GaussianMixture
from sklearn.model_selection import GridSearchCV


class GaussianMixtureBIC:

    logger = logging.getLogger('Debug')

    def __init__(self, max_iterations=100):
        self.data = None
        self.model = None
        self.max_iterations = max_iterations

    def fit(self, X, **kwargs):
        self.data = X
        self.model = self.simple_search_gmm()
        return self

    # TODO Splitting in a training and holdout dataset could be useful to prevent overfitting
    # -> Fit model to training set and evaluate on holdout
    def simple_search_gmm(self, min_comp=1, max_comp=30, steps=2, covariance_type='full'):
        lowest_bic = np.infty
        bic = []

        search_range = range(min_comp, max_comp, steps)
        n_components_range = list(search_range)
        for n_components in n_components_range:
            # Fit a Gaussian mixture with EM
            gmm = mixture.GaussianMixture(n_components=n_components,
                                          covariance_type=covariance_type)
            try:
                gmm.fit(self.data)
                bic.append(gmm.bic(self.data))
                if bic[-1] < lowest_bic:
                    lowest_bic = bic[-1]
                    best_gmm = gmm
                    if n_components == len(n_components_range) - 1 and n_components < self.max_iterations:
                        n_components_range = n_components_range + list(range(n_components, n_components + 10))
                        self.logger.debug("Convergence Warning in gmm. Increase components.")
            except ValueError as e:
                self.logger.error("VALUE ERROR IN GMM. Something didnt work. Check your datasets for invalid types.")
                self.logger.error(traceback._cause_message)
        return best_gmm

    def sample(self, n_samples=1) -> np.ndarray:
        return self.model.sample(n_samples)[0]


class GaussianMixtureCV:

    logger = logging.getLogger('Debug')

    def __init__(self, no_components_list: Union[np.ndarray, List[int]], cv=5):
        self.no_components_list = no_components_list
        self.model = None
        self.cv = cv

    def fit(self, X, **kwargs):
        gmm_paramas = {"n_components": self.no_components_list, "covariance_type": ["full", "tied", "diag", "spherical"]}
        gmm_cv = GridSearchCV(GaussianMixture(), gmm_paramas, cv=self.cv)
        gmm_cv.fit(X)
        self.model = gmm_cv.best_estimator_
        return self

    def sample(self, n_samples=1) -> np.ndarray:
        return self.model.sample(n_samples)[0]