from sklearn.linear_model import LogisticRegression


class Logistic:

    def __init__(self, solver, multi_class, random_state):
        self._instances = None
        self._classes = None
        self._model = LogisticRegression(solver=solver, multi_class=multi_class, random_state=random_state)
        self._trained_model = None
        self._predictions = None

    def train(self, instances, classes):
        self._instances = instances
        self._classes = classes
        self._trained_model = self._model.fit(instances, classes)

    def predict(self, instances):
        self._predictions = self._model.predict(instances)
        return self._predictions
