from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold


class Voting_Classifier:

    def __init__(self):
        self._instances = None
        self._classes = None
        self._model = None
        self._trained_model = None

    def train(self, instances, classes):
        self._instances = instances
        self._classes = classes
        self._trained_model = self._model.fit(instances, classes)

    def predict(self, instances):
        return self._model.predict(instances)

    def create_estimators (self):

        clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
        c2f2 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=2)
        c3f3 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=3)
        c4f4 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=4)
        c5f5 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=5)
        c6f6 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=6)

        estimators = [('lbfgs', clf1), ('lbfgs2', c2f2), ('lbfgs3', c3f3), ('lbfgs4', c4f4), ('lbfgs5', c5f5), ('lbfgs6', c6f6)]
        clasificador = VotingClassifier(estimators)
        self._model = clasificador


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
