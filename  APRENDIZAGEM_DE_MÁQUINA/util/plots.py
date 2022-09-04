from sklearn import metrics
import matplotlib.pyplot as plt
import numpy as np
import itertools


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


def plot_surface_boundary(trainSet, predict, feature_names, target_names):
    print("Surface Boundary")
    plot_colors = "ryb"
    plot_step = 0.02
    n_classes = 2
    x_min, x_max = trainSet[:, 0].min() - 1, trainSet[:, 0].max() + 1
    y_min, y_max = trainSet[:, 1].min() - 1, trainSet[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, plot_step), np.arange(y_min, y_max, plot_step))
    plt.tight_layout(h_pad=0.5, w_pad=0.5, pad=2.5)

    Z = predict.reshape(xx.shape)
    cs = plt.contourf(xx, yy, Z, cmap=plt.cm.RdYlBu)

    plt.xlabel(feature_names[0])
    plt.ylabel(feature_names[1])

    # Plot the training points
    for i, color in zip(range(n_classes), plot_colors):
        idx = np.where(predict == i)
        plt.scatter(trainSet[idx, 0], trainSet[idx, 1], c=color, label=target_names[i], cmap=plt.cm.RdYlBu, edgecolor='black', s=15)

    plt.suptitle("Decision surface of a decision tree using paired features")
    plt.legend(loc='lower right', borderpad=0, handletextpad=0)
    plt.axis("tight")
    plt.show()
