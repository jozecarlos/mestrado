import random

from sklearn import metrics, neighbors, __all__
import matplotlib.pyplot as plt
import numpy as np
import itertools
from dataset import Dataset


def p_confusion_matrix(cm, classes,
                       normalize=False,
                       title='Confusion matrix',
                       cmap=plt.cm.Blues):
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, cm[i, j],
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.show()


def plot_surface_boundary(dataset, type, idx_class):
    import numpy as np
    import pandas as pd
    from sklearn import preprocessing

    arr = np.array(dataset)

    df = pd.DataFrame(arr)
    df_new = df.iloc[:, [0, 1]]
    print(df_new.to_latex())

    D = arr[:, 0:idx_class]
    X1 = D[:, 0:2]
    X = X1.astype(np.float)

    le = preprocessing.LabelEncoder()
    y1 = arr[:, idx_class]
    le.fit(y1)
    y = le.transform(y1)

    # definir limites do domínio
    min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1
    # definir a escala x e y
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # achatar cada grade para um vetor
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    #
    # vetores de pilha horizontal para criar entrada x1,x2 para o modelo
    grid = np.hstack((r1, r2))
    # definir o modelo
    model = neighbors.KNeighborsClassifier(4, weights='uniform')
    if type == 'dmc':
        model = neighbors.NearestCentroid()
    # treinar o modelo
    model.fit(X, y)
    # fazer as predições da grade
    yhat = model.predict(grid)
    # remodelar as previsões de volta em uma grade
    zz = yhat.reshape(xx.shape)
    # plotar a grade de valores x, y e z como uma superfície
    plt.contourf(xx, yy, zz, cmap='Paired')
    #
    # crie um gráfico de dispersão para amostras de cada classe
    for class_value in range(2):
        # obter índices de linha para amostras com esta classe
        row_ix = np.where(y == class_value)
        # criar dispersão dessas amostras
        plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')

    plt.show()
