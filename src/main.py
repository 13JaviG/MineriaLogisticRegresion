import pandas as pd
import numpy as np
import random
from src.tfidf import create_tfidf
from src.bootstrap import bootstrap
from src.logistic import Logistic
from src.results import Resultado
from src.voting_classifier import Voting_Classifier

# Clase principal que ejecuta el programma:
# A partir de un conjunto de datos (autopsias verbales) en formato tf-idf realiza dos clasificaciones:
# -Logistic Regression (con varios bootstrap)
# -Voting Classifier de varios Logistic Regression
# y exporta los resultados (figuras de mérito) a un fichero de textos


def main():
    #Obtener instancias del conjunto de datos
    p = 1.00
    data = pd.read_csv("../data/verbal_autopsies_clean.csv", header=0, skiprows=lambda i: i > 0 and random.random() > p)
    print('Numero de instancias: {}'.format(len(data)))
    clases = np.array(data['gs_text34'])
    clasesunique = np.unique(clases)
    num_classes = len(clasesunique)
    indices = np.array(data['newid'])
    #Convertir a tfidf
    data_tfidf = create_tfidf(data)
    instances_sinclase = np.column_stack((data_tfidf, clases))
    instances = np.column_stack((indices, instances_sinclase))

    # configure bootstrap
    n_iterations = 5
    n_size = int(len(instances))

    # configurar clasificador Logistic
    classifier = Logistic('lbfgs', 'multinomial', 1)

    avg_precision, avg_recall, avg_accuracy, avg_f_score, avg_kappa = 0, 0, 0, 0, 0
    for i in range(n_iterations):
        # Separamos en test y train con bootstrap
        train, test = bootstrap(instances, n_size)

        # fit model
        print("Empieza el entrenamiento del LogisticRegression")
        classifier.train(train[:, :-1], train[:, -1])
        print("Termina el entrenamiento del LogisticRegression")

        # evaluate model
        print("Empieza la evaluación del LogisticRegression")
        trueclasses = test[:, -1]
        predictions = classifier.predict(test[:, :-1])

        # Obtener figuras de méritos
        results = Resultado(trueclasses, predictions)
        avg_precision += results.precision()
        avg_recall += results.recall()
        avg_accuracy += results.accuracy()
        avg_f_score += results.f_score()
        avg_kappa += results.kappa()

        print("Termina la evaluación del LogisticRegression")

    # Exportamos los ficheros de los resultados del Logistic en la carpeta results
    favg_precision = avg_precision / n_iterations
    favg_recall = avg_recall / n_iterations
    favg_accuracy = avg_accuracy / n_iterations
    favg_f_score = avg_f_score / n_iterations
    favg_kappa = avg_kappa / n_iterations

    res_baseline = '==============================================================\n'
    res_baseline += 'MEDIA DE TODAS LAS CLASES MODELO BASELINE (LOGISTIC REGRESSION\n'
    res_baseline += '==============================================================\n'
    res_baseline += 'Precision: \t{}\n'.format(favg_precision)
    res_baseline += 'Recall: \t{}\n'.format(favg_recall)
    res_baseline += 'Accuracy: \t{}\n'.format(favg_accuracy)
    res_baseline += 'F-Score: \t{}\n'.format(favg_f_score)
    res_baseline += 'Kappa: \t\t{}\n'.format(favg_kappa)

    file = open('../results/resultado_baseline.txt', 'w')
    file.write(res_baseline)
    file.close()

    # Clasificación VotingClassifier
    voting = Voting_Classifier()
    rs = Resultado()
    results = voting.k_fold_cross_v(10, data_tfidf, clases, rs)

    #Exportamos los resultados en la carpeta results
    favg_precision_2 = results[0]
    favg_recall_2 = results[1]
    favg_f_score_2 = results[2]
    favg_accuracy_2 = results[3]
    favg_kappa_2 = results[4]
    res_voting = '=============================================================\n'
    res_voting += 'MEDIA DE TODAS LAS CLASES VOTING CLASSIFIER (VARIOS LOGISTIC)\n'
    res_voting += '=============================================================\n'
    res_voting += 'Precision: \t{}\n'.format(favg_precision_2)
    res_voting += 'Recall: \t{}\n'.format(favg_recall_2)
    res_voting += 'Accuracy: \t{}\n'.format(favg_accuracy_2)
    res_voting += 'F-Score: \t{}\n'.format(favg_f_score_2)
    res_voting += 'Kappa: \t\t{}\n'.format(favg_kappa_2)

    file = open('../results/resultado_Voting.txt', 'w')
    file.write(res_voting)
    file.close()


if __name__ == '__main__':
    main()
