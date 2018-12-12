import sklearn.metrics as mt


class Resultado:

    def __init__(self, trueclasses, predictions, clase):
        self._trueclasses = trueclasses
        self._predictions = predictions
        self._clase = clase
        self._TP, self._FP, self._TN, self._FN = self.matriz_confusion(clase)

    def precision(self):
        try:
            #precision = mt.precision_score(self._trueclasses, self._predictions, labels=self._clase, average='weighted')
            precision = 100 * self._TP / (self._TP + self._FP)
        except ZeroDivisionError:
            precision = 0
        return precision

    def recall(self):
        try:
            #recall = mt.recall_score(self._trueclasses, self._predictions, labels=self._clase, average='weighted')
            recall = self.tpr(self._clase)
        except ZeroDivisionError:
            recall = 0
        return recall

    def accuracy(self):
        try:
            #accuracy = mt.accuracy_score(self._trueclasses, self._predictions)
            accuracy = (self._TP + self._TN) / (self._TP + self._FN + self._FP + self._TN)
        except ZeroDivisionError:
            accuracy = 0
        return accuracy

    def f_score(self):
        try:
            #f_score = mt.f1_score(self._trueclasses, self._predictions, labels=self._clase, average='weighted')
            precision = self.precision()
            recall = self.recall()
            f_score = 2 + precision + recall / (precision + recall)
        except ZeroDivisionError:
            f_score = 0
        return f_score

    def kappa(self):
        try:
            kappa = mt.cohen_kappa_score(self._trueclasses, self._predictions, labels=self._clase)
        except ZeroDivisionError:
            kappa = 0
        return kappa

    def tpr(self):
        try:
            #tpr = cm['tp'] / (cm['tp'] + cm['fn'])
            tpr = self._TP / (self._TP + self._FN)
        except ZeroDivisionError:
            tpr = 0
        return tpr

    def fnr(self, clase):
        try:
            #fnr = cm['fn'] / (cm['tp'] + cm['fn'])
            fnr = self._FN / (self._TP + self._FN)
        except ZeroDivisionError:
            fnr = 0
        self._text['FNR'][clase] = fnr
        return fnr

    def fpr(self):
        try:
            #fpr = cm['fp'] / (cm['fp'] + cm['tn'])
            fpr = self._FP / (self._FP + self._TN)
        except ZeroDivisionError:
            fpr = 0
        return fpr

    def tnr(self):
        try:
            #tnr = cm['tn'] / (cm['fp'] + cm['tn'])
            tnr = self._TN / (self._FP + self._TN)
        except ZeroDivisionError:
            tnr = 0
        return tnr

    def matriz_confusion(self, clase):
        TP, FP, FN, TN = 0, 0, 0, 0
        for i in range(len(self._predictions)):
            pred = self._predictions[i]
            real = self._trueclasses[i]
            if pred == real:
                if pred == clase:
                    TP += 1
                else:
                    TN += 1
            else:
                if pred == clase:
                    FP += 1
                else:
                    FN += 1
        return {TP, FP, TN, FN}
