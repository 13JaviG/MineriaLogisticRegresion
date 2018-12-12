from sklearn.linear_model import LogisticRegression

# Clase que crea un clasificador Logistic Regression
class Logistic:

    # Inicializamos el Logistic
    def __init__(self, solver, multi_class, random_state):
        self._instances = None
        self._classes = None
        self._model = LogisticRegression(solver=solver, multi_class=multi_class, random_state=random_state)
        self._trained_model = None
        self._predictions = None

    # Entrenamos el modelo con los datos introducidos
    def train(self, instances, classes):
        self._instances = instances
        self._classes = classes
        self._trained_model = self._model.fit(instances, classes)

    # Predice las clases de un conjunto de datods dado
    def predict(self, instances):
        self._predictions = self._model.predict(instances)
        return self._predictions
