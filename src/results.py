import sklearn.metrics as mt

# Clase que sirve para genera las figuras de mérito de una clasificación


class Resultado:

    def __init__(self, trueclasses=None, predictions=None):
        self._trueclasses = trueclasses
        self._predictions = predictions

    def trueclasses(self, clases):
        self._trueclasses = clases

    def predictions(self, clases):
        self._predictions = clases

    def precision(self):
        try:
            precision = mt.precision_score(self._trueclasses, self._predictions, average='weighted')
        except ZeroDivisionError:
            precision = 0
        return precision

    def recall(self):
        try:
            recall = mt.recall_score(self._trueclasses, self._predictions, average='weighted')
        except ZeroDivisionError:
            recall = 0
        return recall

    def accuracy(self):
        try:
            accuracy = mt.accuracy_score(self._trueclasses, self._predictions)
        except ZeroDivisionError:
            accuracy = 0
        return accuracy

    def f_score(self):
        try:
            f_score = mt.f1_score(self._trueclasses, self._predictions, average='weighted')
        except ZeroDivisionError:
            f_score = 0
        return f_score

    def kappa(self):
        try:
            kappa = mt.cohen_kappa_score(self._trueclasses, self._predictions)
        except ZeroDivisionError:
            kappa = 0
        return kappa
