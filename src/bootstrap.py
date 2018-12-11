from sklearn.utils import resample
import numpy as np

# Realiza un bootstrap de un conjunto de datos y crea los conjuntos train y test


def bootstrap(instances, n_size):
    print("Empieza el Bootstrap")
    boot = resample(instances, replace=True, n_samples=n_size, random_state=None)
    indexes = boot[:, 0]
    test = np.array([x for x in instances if x[0] not in indexes])
    print("Finaliza el Bootstrap")
    return boot[:, 1:], test[:, 1:]
