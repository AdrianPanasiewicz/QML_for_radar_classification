import numpy as np
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

class SupportVectorMachine(BaseEstimator, ClassifierMixin):
    def __init__(
        self,
        *,
        config=None,
        C=1.0,
        kernel="rbf",
        degree=3,
        gamma="scale",
        coef0=0.0,
        shrinking=True,
        probability=False,
        tol=1e-3,
        cache_size=200,
        class_weight=None,
        verbose=False,
        max_iter=-1,
        decision_function_shape="ovr",
        break_ties=False,
        random_state=None,
        use_scaler=True,
        kernel_callable=None
    ):
        self.C = C
        self.kernel = kernel
        self.degree = degree
        self.gamma = gamma
        self.coef0 = coef0
        self.shrinking = shrinking
        self.probability = probability
        self.tol = tol
        self.cache_size = cache_size
        self.class_weight = class_weight
        self.verbose = verbose
        self.max_iter = max_iter
        self.decision_function_shape = decision_function_shape
        self.break_ties = break_ties
        self.random_state = random_state
        self.use_scaler = use_scaler
        self.kernel_callable = kernel_callable
        self.model_name = self.__class__.__name__

    def _resolve_kernel(self):
        if self.kernel == "callable":
            if self.kernel_callable is None:
                raise ValueError("kernel_callable must be provided when kernel='callable'")
            return self.kernel_callable
        return self.kernel

    def fit(self, X, y):
        svc = SVC(
            C=self.C,
            kernel=self._resolve_kernel(),
            degree=self.degree,
            gamma=self.gamma,
            coef0=self.coef0,
            shrinking=self.shrinking,
            probability=self.probability,
            tol=self.tol,
            cache_size=self.cache_size,
            class_weight=self.class_weight,
            verbose=self.verbose,
            max_iter=self.max_iter,
            decision_function_shape=self.decision_function_shape,
            break_ties=self.break_ties,
            random_state=self.random_state
        )

        if self.use_scaler and self.kernel != "precomputed" and self.kernel != "callable":
            self.model_ = make_pipeline(StandardScaler(), svc)
        else:
            self.model_ = svc

        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def decision_function(self, X):
        return self.model_.decision_function(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def score(self, X, y):
        return self.model_.score(X, y)