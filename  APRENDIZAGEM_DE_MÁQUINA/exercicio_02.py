import os
import sys

import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib import pyplot as plt
from sklearn import preprocessing, metrics
from sklearn.datasets import make_blobs
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.utils import shuffle

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from classifier import gaussian_bayes_classifier
from classifier import naive_bayes_classifier


def plot_surface_boundary(dataset, type, title_plot):
    import numpy as np
    import matplotlib.pyplot as plt

    X = dataset.iloc[:, :2]  # we only take the first two features.
    y = dataset.iloc[:, -1]

    def make_meshgrid(x, y, h=.02):
        x_min, x_max = x.min() - 1, x.max() + 1
        y_min, y_max = y.min() - 1, y.max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
        return xx, yy

    def plot_contours(ax, clf, xx, yy, **params):
        Z = clf.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        out = ax.contourf(xx, yy, Z, **params)
        return out

    model = GaussianProcessClassifier()
    if type == 'naive':
        model = GaussianNB()

    clf = model.fit(X, y)

    fig, ax = plt.subplots()
    # title for the plots
    title = (title_plot)
    # Set-up grid for plotting.
    X0, X1 = X.iloc[:, 0], X.iloc[:, 1]
    xx, yy = make_meshgrid(X0, X1)

    plot_contours(ax, clf, xx, yy, cmap=plt.cm.coolwarm, alpha=0.8)
    ax.scatter(X0, X1, c=y, cmap=plt.cm.coolwarm, s=20, edgecolors='k')
    ax.set_ylabel('y')
    ax.set_xlabel('x')
    ax.set_xticks(())
    ax.set_yticks(())
    ax.set_title(title)
    ax.legend()
    plt.show()


def print_interactions(interations, label):
    acc_values = []
    acc_sum = 0
    std_sum = 0

    for i in range(len(interations)):
        acc_values.append(interations[i][4])
        acc_sum = interations[i][4] + acc_sum
        std_sum = interations[i][5] + std_sum

    max_index = acc_values.index(max(acc_values))
    min_index = acc_values.index(min(acc_values))
    print(label)
    print("Melhor Acurácia: " + repr(interations[max_index][4]))
    print("Pior Acurácia: " + repr(interations[min_index][4]))
    print("Std Max: " + repr(interations[max_index][5]))
    print("Std Min: " + repr(interations[min_index][5]))
    print('Accuracy Mean: ' + repr(acc_sum / 20) + '%')
    print('Std Mean: ' + repr(std_sum / 20))
    print("------------------------------------------------")

    confusion_matrix = interations[max_index][6]
    cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix)
    cm_display.plot()
    plt.show()


def GaussianByesClassifier(dataset, idx_class):
    (x_train, x_test, y_train, y_test) = \
        train_test_split(dataset.iloc[:, 0:idx_class].values, dataset.iloc[:, idx_class].values, train_size=0.8)
    mlc = gaussian_bayes_classifier.MLClassifier()
    mlc.fit(x_train, y_train)

    predictions = np.array([mlc.predict(x_test[i]) for i in range(x_test.shape[0])])
    accuracy = mlc.accuracy(predictions, y_test)
    std = np.std(x_test)
    confusion_matrix = metrics.confusion_matrix(y_test, predictions)

    return [x_train, x_test, y_train, y_test, accuracy, std, confusion_matrix]


def NaivesBayesClassifier(dataset, idx_class):
    sh_df = shuffle(dataset)
    (x_train, x_test, y_train, y_test) = \
        train_test_split(sh_df.iloc[:, 0:idx_class].values, sh_df.iloc[:, idx_class].values, train_size=0.8)
    naives = naive_bayes_classifier.NaiveBayes()
    naives.fit(x_train, y_train)

    predictions = naives.predict(x_test)
    accuracy = naives.accuracy(predictions, y_test)
    std = np.std(x_test)
    confusion_matrix = metrics.confusion_matrix(y_test, predictions)

    return [x_train, x_test, y_train, y_test, accuracy, std, confusion_matrix]


def process_data(dt, idx_class, lb=True, selected_colums=[]):
    print("Iris DataSet Process")

    if isinstance(dt, pd.DataFrame):
        df = dt
    else:
        df = pd.read_csv(dt)

    columns = df.columns.tolist()

    for column in df:
        if df[column].isnull().any():
            df[column].fillna(df[column].mean(), inplace = True)

    d = preprocessing.normalize(df.iloc[:, :-1])
    scaled_df = pd.DataFrame(d, columns=columns[0:-1])

    scaled_df[columns[-1]] = df[columns[-1]]
    df = shuffle(scaled_df)

    if lb:
        label_encoder = preprocessing.LabelEncoder()
        df.iloc[:, -1] = label_encoder.fit_transform(df.iloc[:, -1:].values.ravel())

    mle_interactions = []
    naive_interactions = []

    for n in range(19):
        mle_interactions.append(GaussianByesClassifier(df, idx_class))
        naive_interactions.append(NaivesBayesClassifier(df, idx_class))

    print_interactions(mle_interactions, "Classificador Bayesiano Gaussiano")
    print_interactions(naive_interactions, "Classificador Naive Bayes")

    for c in selected_colums[0]:
        sns.displot(data=df, x=c, hue=selected_colums[1], kind='kde', fill=True,
                    palette=sns.color_palette('bright')[:3], height=5, aspect=1.5)
        plt.show()

    print("Finished Data Process")
    return df


def create_artificial_data(columns):
    dataset = []
    classes = ['Terraqueo', 'Marciano', 'Jupiteriano']
    X, y = make_blobs(n_samples=100, centers=3, cluster_std=1, n_features=2)
    for i in range(len(X)):
        el = X[i].tolist()
        el.append(classes[y[i]])
        dataset.append(el)

    data_frame = pd.DataFrame(dataset, columns=columns)

    return data_frame


if __name__ == "__main__":
    dataset_iris = process_data('./datasets/iris_dataset.csv', 4, True,
                                [['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)'],
                                 'target'])
    print(dataset_iris.head(n=20).to_latex(index=False))
    plot_surface_boundary(dataset_iris, "gaus", "Superfície de Decisão Iris")
    print('-------------------')
    print('-------------------')
    print('-------------------')
    coluna_vertebral = process_data('./datasets/output.csv', 6, True,
                                    [['pelvic_incidence', 'pelvic_tilt', 'lumbar_lordosis_angle', 'sacral_slope'],
                                     'class'])
    print(coluna_vertebral.head(n=20).to_latex(index=False))
    plot_surface_boundary(coluna_vertebral, "gaus", "Superfície de Decisão Coluna Vertebral")
    print('-------------------')
    print('-------------------')
    print('-------------------')
    dermatology = process_data('./datasets/dermatology.csv', 34, True,
                               [['erythema', 'scaling', 'definite_borders', 'itching'], 'class'])
    print(dermatology.iloc[:,:4].head(n=20).to_latex(index=False))
    plot_surface_boundary(dermatology, "gaus", "Superfície de Decisão Dermatology")
    print('-------------------')
    print('-------------------')
    print('-------------------')
    breast_cancer = process_data('./datasets/breast_cancer.csv', 5, True,
                                 [['mean_radius', 'mean_texture', 'mean_perimeter', 'mean_area'], 'diagnosis'])
    print(breast_cancer.head(n=20).to_latex(index=False))
    plot_surface_boundary(breast_cancer, "gaus", "Superfície de Decisão Breast Cancer")
    print('-------------------')
    print('-------------------')
    print('-------------------')
    dataset = create_artificial_data(['feature 1', 'feature 2', 'class'])
    print(dataset.head().to_latex(index=False))
    dataset = process_data(dataset, 2, True,
                 [['feature 1', 'feature 2'], 'class'])
    print(dataset.head(n=20).to_latex(index=False))
    plot_surface_boundary(dataset, "gaus", "Superfície de Decisão Artificial 01")
