import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import sys


def pca_graphic(data_tfidf, clases, indexes):

    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(data_tfidf)
    principal = np.column_stack(( principalComponents, clases))
    principal1 = np.column_stack((indexes, principal))
    principalDf = pd.DataFrame(data=principal1,index = indexes, columns=['index','principal component 1', 'principal component 2','gs_text34'])

    x1 = np.array(principalDf['principal component 1'])
    x2 = np.array(principalDf['principal component 2'])
    x3 = np.array(principalDf['gs_text34'])
    X = np.column_stack((x1, x2))
    Y = np.column_stack((X, x3))
    x_min, x_max = float(X[:, 0].min()) - .5, float(X[:, 0].max()) + .5
    y_min, y_max = float(X[:, 1].min()) - .5, float(X[:, 1].max()) + .5
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = X
    # Put the result into a color plot
    #Z = Z.reshape(xx.shape)
    plt.figure(figsize=(10, 10))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()
    sys.exit()

    plt.figure(figsize=(10, 10))
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('2 component PCA')
    #ax.set_xlabel('Principal Component 1', fontsize=15)
    #ax.set_ylabel('Principal Component 2', fontsize=15)
    #ax.set_title('2 component PCA', fontsize=20)



    targets = np.unique(clases)

    for target in targets:
        color = np.random.rand(1,3)
        indicesToKeep =  principalDf['gs_text34'] == target
        plt.scatter( principalDf.loc[indicesToKeep, 'principal component 1'],  principalDf.loc[indicesToKeep, 'principal component 2'], c= color, edgecolors= color, cmap=plt.cm.Paired, s=50)
    plt.legend(targets)
    plt.grid()
    plt.show()

    x_min, x_max = X[:, 0].min() - .5, X[:, 0].max() + .5
    y_min, y_max = X[:, 1].min() - .5, X[:, 1].max() + .5
    h = .02  # step size in the mesh
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
    Z = logreg.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(4, 3))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)


    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.show()
