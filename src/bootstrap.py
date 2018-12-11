from sklearn.utils import resample
import numpy as np


def bootstrap(instances, n_size):
    # prepare train and test sets
    print("Empieza el Bootstrap")
    boot = resample(instances, replace=True, n_samples=n_size, random_state=None)
    indexes = boot[:, 0]
    test = np.array([x for x in instances if x[0] not in indexes])
    #test = np.array([x for x in instances if x.tolist() not in boot.tolist()])
    print("Finaliza el Bootstrap")
    return boot[:, 1:], test[:, 1:]
