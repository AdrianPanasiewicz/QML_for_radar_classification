from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.svm import SVC

class QuantumSupportVectorMachine(BaseEstimator, ClassifierMixin):
    def __init__(self,C=1.0, class_weight=None, encoding=None):
        self.C = C
        self.encoding = encoding

    def fit(self, X, y):
        self.model_ = SVC(
            C=self.C,
            kernel='precomputed'
        )
        self.model_.fit(X, y)
        return self

    def predict(self, X):
        return self.model_.predict(X)

    def predict_proba(self, X):
        return self.model_.predict_proba(X)

    def decision_function(self, X):
        return self.model_.decision_function(X)

    def score(self, X, y):
        return self.model_.score(X, y)