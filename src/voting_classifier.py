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

        clf1 = LogisticRegression(solver='sag', multi_class='multinomial', random_state=1)
        c2f2 = LogisticRegression(solver='saga', multi_class='multinomial', random_state=1)
        c3f3 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
        c4f4 = LogisticRegression(solver='sag', multi_class='multinomial', random_state=3)
        c5f5 = LogisticRegression(solver='saga', multi_class='multinomial', random_state=3)
        c6f6 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=3)

        estimators = [('sag', clf1), ('saga', c2f2), ('lbfgs', c3f3), ('sag1', c4f4), ('saga1', c5f5), ('lbfgs1', c6f6)]
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
