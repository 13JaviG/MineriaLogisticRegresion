import pandas as pd
import numpy as np
import random
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from matplotlib import pyplot


def main():
    p = 1.00
    data = pd.read_csv("../data/verbal_autopsies_clean.csv", header=0, skiprows=lambda i: i > 0 and random.random() > p)
    print('Numero de instancias: {}'.format(len(data)))
    clases = np.array(data['gs_text34'])

    data_tfidf = create_tfidf(data)

    instances = np.column_stack((data_tfidf, clases))

    #lab_enc = preprocessing.OrdinalEncoder()
    #instances = lab_enc.fit_transform(info)

    # configure bootstrap
    n_iterations = 10
    n_size = int(len(instances) * 0.632)

    # run bootstrap
    stats = list()

    #Hacemos el Bootstrap#
    #train, test = bootstrap(instances, n_size)
    for i in range(n_iterations):
    # Separamos en test y train
        train, test = train_test_split(instances, test_size=0.368)
        train = bootstrap_single(train, len(train))
        test = bootstrap_single(test, len(test))

        # fit model
        print("Empieza el entrenamiento del LogisticRegression")
        model = LogisticRegression(solver='lbfgs', multi_class='multinomial')
        model.fit(train[:, :-1], train[:, -1])
        print("Termina el entrenamiento del LogisticRegression")

        # evaluate model
        print("Empieza la evaluación del LogisticRegression")
        predictions = model.predict(test[:, :-1])
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


def create_tfidf(data_frame):

    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_frame['open_response'].values.astype('U'))

    return tfidf_matrix.A


def bootstrap(instances, n_size):
    
    # prepare train and test sets
    print("Empieza el Bootstrap")
    boot = resample(instances, replace=True, n_samples=n_size, random_state=1)
    test = np.array([x for x in instances if x.tolist() not in boot.tolist()])
    print("Finaliza el Bootstrap")

    return boot, test


def bootstrap_single(instances, n_size):
    # prepare train and test sets
    print("Empieza el Bootstrap_single")
    boot = resample(instances, replace=True, n_samples=n_size, random_state=1)
    print("Finaliza el Bootstrap_single")

    return boot

if __name__ == '__main__':
    main()
