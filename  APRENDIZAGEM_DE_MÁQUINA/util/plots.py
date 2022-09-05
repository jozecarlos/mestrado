from sklearn import metrics
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


def get_classes(data, class_index, unique=None):
    labels = []

    for x in range(len(data)):
        labels.append(data[x][class_index])

    if unique is None:
        return labels
    elif unique:
        return Collection.unique(labels)
    else:
        return labels


def test():
    import numpy as np
    from sklearn.datasets import make_blobs
    from sklearn.linear_model import LogisticRegression

    # generate dataset
    X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=3)
    # define bounds of the domain
    min1, max1 = X[:, 0].min()-1, X[:, 0].max()+1
    min2, max2 = X[:, 1].min()-1, X[:, 1].max()+1
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1,r2))
    # define the model
    model = LogisticRegression()
    # fit the model
    model.fit(X, y)
    # make predictions for the grid
    yhat = model.predict(grid)
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz, cmap='Paired')
    # create scatter plot for samples from each class
    for class_value in range(2):
        # get row indexes for samples with this class
        row_ix = np.where(y == class_value)
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')

    plt.show()

def plot_surface_boundary(dataset, model, idx_class):
    print("Surface Boundary")


    arr = np.array(dataset)
    D = arr[:, 0:idx_class]
    X1 = D[:,0:2]
    X = X1.astype(np.float)
    y1 = arr[:, idx_class]

    #from sklearn.datasets import make_blobs
    #X, y = make_blobs(n_samples=1000, centers=2, n_features=2, random_state=1, cluster_std=3)

    # define bounds of the domain
    min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1
    # define the x and y scale
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # create all of the lines and rows of the grid
    xx, yy = np.meshgrid(x1grid, x2grid)
    # flatten each grid to a vector
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))
    # horizontal stack vectors to create x1,x2 input for the model
    grid = np.hstack((r1, r2))
    # define the model
    #model = LogisticRegression()
    #neighbors = model.train(trainingSet, testSet[x], k)
    #result = model.predict(neighbors)
    #predictions.append(result)
    # fit the model
    model.fit(X, y)
    # make predictions for the grid
    yhat = model.predict(grid)
    # reshape the predictions back into a grid
    zz = yhat.reshape(xx.shape)
    # plot the grid of x, y and z values as a surface
    plt.contourf(xx, yy, zz, cmap='Paired')
    # create scatter plot for samples from each class
    for class_value in range(2):
        # get row indexes for samples with this class
        row_ix = np.where(y == class_value)
        # create scatter of these samples
        plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
