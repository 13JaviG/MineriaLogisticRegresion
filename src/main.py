import pandas as pd
import numpy as np
import random
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from matplotlib import pyplot
from src.tfidf import create_tfidf
from src.bootstrap import bootstrap
from src.logistic import Logistic


def main():
    p = 1.00
    data = pd.read_csv("../data/verbal_autopsies_clean.csv", header=0, skiprows=lambda i: i > 0 and random.random() > p)
    print('Numero de instancias: {}'.format(len(data)))
    clases = np.array(data['gs_text34'])
    indices = np.array(data['newid'])

    print("Empieza TFIDF")
    data_tfidf = create_tfidf(data)
    instances = np.column_stack((data_tfidf, clases))
    instances = np.column_stack((indices, instances))

    # configure bootstrap
    n_iterations = 6
    n_size = int(len(instances))

    # run bootstrap
    stats = list()
    classifier = Logistic('lbfgs', 'multinomial', 1)

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
        matriz_confusion = confusion_matrix(test[:, -1], predictions)
        print(matriz_confusion)
        score = accuracy_score(test[:, -1], predictions)
        print(score)
        stats.append(score)
        print("Termina la evaluación del LogisticRegression")

    # plot scores
    pyplot.hist(stats)
    pyplot.show()
    # confidence intervals
    alpha = 0.95
    p = ((1.0 - alpha) / 2.0) * 100
    lower = max(0.0, np.percentile(stats, p))
    p = (alpha + ((1.0 - alpha) / 2.0)) * 100
    upper = min(1.0, np.percentile(stats, p))
    print('%.1f confidence interval %.1f%% and %.1f%%' % (alpha * 100, lower * 100, upper * 100))


if __name__ == '__main__':
    main()
