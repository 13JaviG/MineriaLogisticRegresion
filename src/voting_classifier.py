from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
import numpy as np
from sklearn.model_selection import KFold


class VotingClassifier():

    def __init__(self, **kwargs):
        self._instances = None
        self._classes = None
        estimators = self.create_estimators()
        self._model = VotingClassifier(estimators, voting='hard')
        self._trained_model = None
        self._kwargs = kwargs



    def train(self, instances, classes):
        self._instances = instances
        self._classes = classes
        self._trained_model = self._model.fit(instances, classes)


    def predict(self, instances):
        pass


    def create_estimators (self):


        clf1 = LogisticRegression(solver='sag'  ,multi_class='multinomial', random_state=1)
        c2f2 = LogisticRegression(solver='saga', multi_class='multinomial', random_state=1)
        c3f3 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)

        estimators = [clf1, c2f2, c3f3]

        return estimators

    @staticmethod
    def k_fold_cross_v(k, instances, classes):

        X = instances
        y = classes
        kf = KFold(n_splits=k, shuffle=True)

        for train_index, test_index in kf.split(X):

            X_train, X_test = X[train_index], X[test_index]
            voting_model = VotingClassifier()
            voting_model.train(X_train,y)
            predict = voting_model.predict(X_test)

        return predict