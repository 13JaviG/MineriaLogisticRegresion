from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import VotingClassifier
from sklearn.model_selection import KFold
from src.results import Resultado
import numpy as np


class Voting_Classifier:

    def __init__(self):
        self._instances = None
        self._classes = None
        self._model = None
        self._trained_model = None
        self._predictions = None
        self.create_estimators()

    def train(self, instances, classes):
        self._instances = instances
        self._classes = classes
        self._trained_model = self._model.fit(instances, classes)

    def predict(self, instances):
        self._predictions = self._model.predict(instances)
        return self._predictions

    def create_estimators(self):

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
    def k_fold_cross_v(k, instances, classes, rs):

        X = instances
        y = classes
        kf = KFold(n_splits=k, shuffle=True)

        precision_ar = []
        recall_ar = []
        f_score_ar =  []
        accuracy_ar = []

        predict = {}

        for train_index, test_index in kf.split(X):
            X_train, X_test = X[train_index], X[test_index]
            voting_model = VotingClassifier()
            voting_model.train(X_train, y)
            predict = voting_model.predict(X_test)

            #Cambiamos los atributos del objeto Results
            rs.trueclasses(X_test[:,-1])
            rs.trueclasses(predict)

            #Guardamos los resultados de cada iteración en un array
            precision_ar = precision_ar.append(rs.precision())
            recall_ar = recall_ar.append(rs.recall())
            f_score_ar = f_score_ar.append(rs.f_score())
            accuracy_ar = accuracy_ar.append(rs.accuracy())

        #Hacemos la media de todos los resultados
        precision = np.mean(precision_ar)
        recall = np.mean(recall_ar)
        f_score = np.mean(f_score_ar)
        accuracy = np.mean(accuracy_ar)

        resultados = [precision, recall, f_score, accuracy]

        return resultados

    def _results_to_text(self, indiv_results):
        text_results = {}
        num_results = len(indiv_results)
        class_names = np.unique(self._classes)
        num_classes = len(class_names)
        cum_precision, cum_recall, cum_accuracy, cum_f_score, cum_kappa, cum_tpr, cum_fnr, cum_fpr, cum_tnr = 0, 0, 0, 0, 0, 0, 0, 0, 0
        for class_name in class_names:
            class_cum_precision, class_cum_recall, class_cum_accuracy, class_cum_f_score, class_cum_kappa, class_cum_tpr, class_cum_fnr, class_cum_fpr, class_cum_tnr = 0, 0, 0, 0, 0, 0, 0, 0, 0
            for results in indiv_results:
                class_cum_precision += results.precision(class_name)
                class_cum_recall += results.recall(class_name)
                class_cum_accuracy += results.accuracy(class_name)
                class_cum_f_score += results.f_score(class_name)
                class_cum_kappa += results.kappa(class_name)
                class_cum_tpr += results.tpr(class_name)
                class_cum_fnr += results.fnr(class_name)
                class_cum_fpr += results.fpr(class_name)
                class_cum_tnr += results.tnr(class_name)

            class_avg_precision = class_cum_precision / num_results
            class_avg_recall = class_cum_recall / num_results
            class_avg_accuracy = class_cum_accuracy / num_results
            class_avg_f_score = class_cum_f_score / num_results
            class_avg_kappa = class_cum_kappa / num_results
            class_avg_tpr = class_cum_tpr / num_results
            class_avg_fnr = class_cum_fnr / num_results
            class_avg_fpr = class_cum_fpr / num_results
            class_avg_tnr = class_cum_tnr / num_results

            cum_precision += class_avg_precision
            cum_recall += class_avg_recall
            cum_accuracy += class_avg_accuracy
            cum_f_score += class_avg_f_score
            cum_kappa += class_avg_kappa
            cum_tpr += class_avg_tpr
            cum_fnr += class_avg_fnr
            cum_fpr += class_avg_fpr
            cum_tnr += class_avg_tnr

            tmp_text = '=========================\n'
            tmp_text += 'MEDIAS DE LA CLASE: {}\n'.format(class_name.upper())
            tmp_text += '=========================\n'
            tmp_text += 'Precision: \t{}\n'.format(class_avg_precision)
            tmp_text += 'Recall: \t{}\n'.format(class_avg_recall)
            tmp_text += 'Accuracy: \t{}\n'.format(class_avg_accuracy)
            tmp_text += 'F-Score: \t{}\n'.format(class_avg_f_score)
            tmp_text += 'Kappa: \t\t{}\n'.format(class_avg_kappa)
            tmp_text += 'TPR: \t\t{}\n'.format(class_avg_tpr)
            tmp_text += 'FNR: \t\t{}\n'.format(class_avg_fnr)
            tmp_text += 'FPR: \t\t{}\n'.format(class_avg_fpr)
            tmp_text += 'TNR: \t\t{}\n'.format(class_avg_tnr)

            text_results[class_name] = tmp_text

        avg_precision = cum_precision / num_classes
        avg_recall = cum_recall / num_classes
        avg_accuracy = cum_accuracy / num_classes
        avg_f_score = cum_f_score / num_classes
        avg_kappa = cum_kappa / num_classes
        avg_tpr = cum_tpr / num_classes
        avg_fnr = cum_fnr / num_classes
        avg_fpr = cum_fpr / num_classes
        avg_tnr = cum_tnr / num_classes

        avg_text = '=========================\n'
        avg_text += 'MEDIA DE TODAS LAS CLASES\n'
        avg_text += '=========================\n'
        avg_text += 'Precision: \t{}\n'.format(avg_precision)
        avg_text += 'Recall: \t{}\n'.format(avg_recall)
        avg_text += 'Accuracy: \t{}\n'.format(avg_accuracy)
        avg_text += 'F-Score: \t{}\n'.format(avg_f_score)
        avg_text += 'Kappa: \t\t{}\n'.format(avg_kappa)
        avg_text += 'TPR: \t\t{}\n'.format(avg_tpr)
        avg_text += 'FNR: \t\t{}\n'.format(avg_fnr)
        avg_text += 'FPR: \t\t{}\n'.format(avg_fpr)
        avg_text += 'TNR: \t\t{}\n'.format(avg_tnr)

        final_text = avg_text
        for class_name in text_results:
            final_text += '\n{}'.format(text_results[class_name])
        return final_text


