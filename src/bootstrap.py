from sklearn.utils import resample
import numpy as np


def bootstrap(instances, n_size):
    # prepare train and test sets
    print("Empieza el Bootstrap")
    boot = resample(instances, replace=True, n_samples=n_size, random_state=None)
    #test = np.setdiff1d(instances.tolist(), boot.tolist())
    test = np.array([x for x in instances if x.tolist() not in boot.tolist()])
    print("Finaliza el Bootstrap")

    return boot, test


def bootstrap_single(instances, n_size):
    # prepare train and test sets
    print("Empieza el Bootstrap_single")
    boot = resample(instances, replace=True, n_samples=n_size, random_state=None)
    print("Finaliza el Bootstrap_single")

    return boot
