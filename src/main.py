import pandas as pd
import numpy as np
import random
import sklearn.metrics as mt
from matplotlib import pyplot
from src.tfidf import create_tfidf
from src.bootstrap import bootstrap
from src.logistic import Logistic
from src.results import Resultado
import sys


def main():
    #Obtener instancias del conjunto de datos
    p = 0.20
    data = pd.read_csv("../data/verbal_autopsies_clean.csv", header=0, skiprows=lambda i: i > 0 and random.random() > p)
    print('Numero de instancias: {}'.format(len(data)))
    clases = np.array(data['gs_text34'])
    clasesunique = np.unique(clases)
    indices = np.array(data['newid'])

    data_tfidf = create_tfidf(data)
    instances = np.column_stack((data_tfidf, clases))
    instances = np.column_stack((indices, instances))

    # configure bootstrap
    n_iterations = 5
    n_size = int(len(instances))

    # configurar clasificador Logistic
    classifier = Logistic('lbfgs', 'multinomial', 1)

    accuracylist = list()

    for i in range(n_iterations):
        # Separamos en test y train con bootstrap
        train, test = bootstrap(instances, n_size)

        # fit model
        print("Empieza el entrenamiento del LogisticRegression")
        classifier.train(train[:, :-1], train[:, -1])
        print("Termina el entrenamiento del LogisticRegression")

        # evaluate model
        print("Empieza la evaluación del LogisticRegression")
        predictions = classifier.predict(test[:, :-1])

        accuracy = mt.accuracy_score(test[:, -1], predictions)
        accuracylist.append(accuracy)
        print("Termina la evaluación del LogisticRegression")

    # plot scores
    pyplot.hist(accuracylist)
    pyplot.show()
    # confidence intervals
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(accuracylist, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(accuracylist, p))
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha * 100, lower * 100, upper * 100))

    resultsPath = '../results/resultado_Baseline.txt'


if __name__ == '__main__':
    main()
