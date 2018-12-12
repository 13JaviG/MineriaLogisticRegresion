from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold

# Este clasificador realiza una combinación paralela por votación de disintos modelos Logistic Regression
class Voting_Classifier:

    def __init__(self):
        self._instances = None
        self._classes = None
        self._model = None
        self._trained_model = None
        self._predictions = None

    def train(self, instances, classes):
        self._instances = instances
        self._classes = classes
        self._trained_model = self._model.fit(instances, classes)

    def predict(self, instances):
        self._predictions = self._model.predict(instances)
        return self._predictions

    def create_estimators(self):
        # Establecemos 6 diferentes modelos del Logistic Regression para realizar la comparativa
        clf1 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=1)
        c2f2 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=2)
        c3f3 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=3)
        c4f4 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=4)
        c5f5 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=5)
        c6f6 = LogisticRegression(solver='lbfgs', multi_class='multinomial', random_state=6)

        estimators = [('lbfgs', clf1), ('lbfgs2', c2f2), ('lbfgs3', c3f3), ('lbfgs4', c4f4), ('lbfgs5', c5f5), ('lbfgs6', c6f6)]
        clasificador = VotingClassifier(estimators)
        self._model = clasificador

    def k_fold_cross_v(self, k, instances, classes, rs):
        X = instances
        y = classes
        kf = KFold(n_splits=k, shuffle=True)

        precision_ar = 0
        recall_ar = 0
        f_score_ar = 0
        accuracy_ar = 0
        kappa_ar = 0
        # Empezamos las iteraciones del k-fold
        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            y_train, y_test = y[train_index], y[test_index]
            self.create_estimators()
            self.train(X_train, y_train)
            predict = self.predict(X_test)

            # Cambiamos los atributos (clases reales y predicciones) del objeto Results
            rs.trueclasses(y_test)
            rs.predictions(predict)

            # Guardamos los resultados de cada iteración en un array
            precision_ar += rs.precision()
            recall_ar += rs.recall()
            f_score_ar += rs.f_score()
            accuracy_ar += rs.accuracy()
            kappa_ar += rs.kappa()

        # Hacemos la media de todos los resultados
        avg_precision = precision_ar / k
        avg_recall = recall_ar / k
        avg_f_score = f_score_ar / k
        avg_accuracy = accuracy_ar / k
        avg_kappa = kappa_ar / k

        resultados = [avg_precision, avg_recall, avg_f_score, avg_accuracy, avg_kappa]
        return resultados
