import csv
import os
import random
import re
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from IPython.display import clear_output
from PIL import Image
from imblearn.metrics import sensitivity_score
from imblearn.metrics import specificity_score
from keras.applications.densenet import DenseNet121
from keras.applications.densenet import DenseNet169
from keras.applications.densenet import DenseNet201
from keras.applications.densenet import preprocess_input as ppi_densenet
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_resnet_v2 import preprocess_input as ppi_inceptionresnet_v2
from keras.applications.inception_v3 import InceptionV3
from keras.applications.inception_v3 import preprocess_input as ppi_inception_v3
from keras.applications.mobilenet import MobileNet
from keras.applications.mobilenet import preprocess_input as ppi_mobilenet
from keras.applications.nasnet import NASNetLarge
from keras.applications.nasnet import NASNetMobile
from keras.applications.nasnet import preprocess_input as ppi_nasnet
from keras.applications.resnet import ResNet50
from keras.applications.resnet import preprocess_input as ppi_resnet50
from keras.applications.vgg16 import VGG16
from keras.applications.vgg16 import preprocess_input as ppi_vgg16
from keras.applications.vgg19 import VGG19
from keras.applications.vgg19 import preprocess_input as ppi_vgg19
from keras.applications.xception import Xception
from keras.applications.xception import preprocess_input as ppi_xception
from scipy.stats import randint as sp_randint
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import *
from sklearn.model_selection import RandomizedSearchCV
from sklearn.naive_bayes import MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier

current_milli_time = lambda: int(round(time.time() * 1000))


# Build the CNN model, without the last fully-connected layers
def build_model(model_name, pooltype='max'):

    model = MobileNet(weights='imagenet', pooling=pooltype, input_shape=(224, 224, 3), include_top=False)

    if 'ResNet50' == model_name:
        model = ResNet50(weights='imagenet', pooling=pooltype, include_top=False)
    if 'DenseNet201' == model_name:
        model = DenseNet201(weights='imagenet', pooling=pooltype, include_top=False)

    return model, 224


def extract_deep_features(file_id, model, model_name, target_size, log=False):
    x = load_img(file_id, target_size)
    x = np.expand_dims(x, axis=0)

    if model_name == 'ResNet50':
        x = ppi_resnet50(x)
    if model_name == 'MobileNet':
        x = ppi_mobilenet(x)
    if model_name == 'DenseNet201':
        x = ppi_densenet(x)

    time_start = current_milli_time()
    features = model.predict(x)
    process_time = current_milli_time() - time_start

    if (log): print('Load process: Complete', current_milli_time() - time_start)

    return features, process_time, x


def load_img(filepath, target_size):
    if filepath.endswith('jpeg'):
        img = cv2.imread(filepath)
    elif filepath.endswith('txt'):
        img = np.loadtxt(filepath).copy()
    elif filepath.endswith('npy'):
        img = np.load(filepath).copy()
    elif filepath.endswith('png') or filepath.endswith('PNG'):
        img = cv2.imread(filepath)

    return cv2.resize(img, (target_size, target_size), interpolation=cv2.INTER_NEAREST)


def deep_extractor(model_name, classes_list, output_fmt=['npy'], srcPath='./data', outPath='./out'):
    print(model_name)
    cont = 0
    csv_aux = 0

    model, targetSize = build_model(model_name)
    data = []
    process_time_list = []

    # Start chronometer
    time_start = current_milli_time()
    data_df_csv = pd.DataFrame()
    # Range: classes
    for fileIdx in classes_list:
        folder_path = srcPath + os.sep + str(fileIdx)
        print(folder_path)
        ds_store_file_location = folder_path + '/.DS_store'
        if os.path.isfile(ds_store_file_location):
            os.remove(ds_store_file_location)
        # For each sample

        for (subdir, dirs, files) in os.walk(folder_path, topdown=True):
            for idx, name in enumerate(files):
                file_path = subdir + os.sep + name
                cont = cont + 1
                if ImagesID:
                    id_img = name.split('_')[0]

                clear_output()
                print(str(cont) + ' | ' + file_path + ' | ' + model_name)

                features, process_time, processed_img = extract_deep_features(file_path, model, model_name, targetSize)
                process_time_list.append(process_time)

                # CSV file has a advantage: it can save the name, or ID, of image (string type)

                if csv_aux == 0:
                    features_df = np.append(features, fileIdx)
                    features_df = np.expand_dims(features_df, axis=0)
                    data_df = pd.DataFrame(data=features_df)
                    image_name = pd.DataFrame(data=np.array([name]))
                    data_df_csv = pd.concat([image_name, data_df], axis=1)
                    csv_aux += 1
                else:
                    features_df2 = np.append(features, fileIdx)
                    features_df2 = np.expand_dims(features_df2, axis=0)
                    data_df = pd.DataFrame(data=features_df2)
                    image_name = pd.DataFrame(data=np.array([name]))
                    data_df = pd.concat([image_name, data_df], axis=1)
                    data_df_csv = pd.concat([data_df_csv, data_df], axis=0)


                # The first column of each line is the image id, the last column is the class of image
                features = features.reshape(-1)
                features = np.hstack((features, fileIdx))
                if ImagesID:
                    features = np.hstack((int(id_img), features))
                data.append(features)

    if 'csv' in output_fmt:
        data_df_csv.to_csv(outPath + os.sep + model_name + '.csv')

    # Save process time into a file
    np.savetxt(outPath + os.sep + model_name + '_time.txt', process_time_list, fmt="%d")

    return sum(list(filter(lambda x: x, process_time_list)))


# Split data per class
def split(ds, classes):
    dsx = []
    for class_id in classes:
        dsi = ds[ds[:, -1] == class_id]
        dsx.append(dsi)
    return dsx


def autolabel(rects, ax):
    """Attach a text label above each bar in *rects*, displaying its height."""
    for rect in rects:
        height = rect.get_height()
        ax.annotate('{0:.2f}'.format(height),
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom')


# Load data for train-test from dataset INBreast
def load_train_test(dataset, ntr, nte, classes):
    list_ds = split(dataset, classes)

    train = []
    test = []

    for i, dsi in enumerate(list_ds):
        if i == 0:
            train = dsi[:ntr[i], :]
            test = dsi[ntr[i]:ntr[i] + nte[i], :]
        else:
            train = np.concatenate((train, dsi[:ntr[i], :]), axis=0)
            test = np.concatenate((test, dsi[ntr[i]:ntr[i] + nte[i], :]), axis=0)

    X_train = np.array(train[:, :-1])
    y_train = np.array(list(map(int, train[:, -1])))
    X_test = np.array(test[:, :-1])
    y_test = np.array(list(map(int, test[:, -1])))

    return X_train, y_train, X_test, y_test


# Normalize data
def normalize_x(X):
    X = np.array(X - np.mean(X))
    X = np.array(((X - np.min(X)) / (np.max(X) - np.min(X))))
    return X


# Shuffle data
def shuffle(x_data, y_data):
    c = list(zip(x_data, y_data))
    random.shuffle(c)
    X, y = zip(*c)
    X = np.array(X)
    y = np.array(y)
    return X, y


def plot_result(model_name_list, classifiers_name_list):
    for model_name in model_name_list:
        plot_csv = pd.read_csv(outPathC + os.sep + 'metrics' + os.sep + model_name + '_mean_std.csv', sep=',')

        x_pos = np.arange(len(plot_csv))
        x_pos = np.arange(1, 2 * x_pos.shape[0] + 1, 2)

        n_classifiers = len(classifiers_name_list)

        width = 0.15

        lines = list(range(len(plot_csv)))

        means = [1, 2, 3, 4, 5, 6]
        stds = [8, 9, 10, 11, 12, 13]

        CTEs = []
        errors = []

        # Metrics
        for i in range(len(metrics_name_list)):
            CTE = [plot_csv.iloc[lines[j], means[i]] for j in range(n_classifiers)]
            error = [plot_csv.iloc[lines[j], stds[i]] for j in range(n_classifiers)]

            CTEs.append(CTE)
            errors.append(error)

        # Plotting
        fig, ax = plt.subplots(figsize=(14, 7))

        r1 = ax.bar(x_pos - width - 0.3, CTEs[0], yerr=errors[0], width=width, align='center', alpha=0.5,
                    ecolor='black',
                    capsize=3, label='Acurácia')
        r2 = ax.bar(x_pos - 0.3, CTEs[1], yerr=errors[1], width=width, align='center', alpha=0.5, ecolor='black',
                    capsize=3, label='Acurácia Balanceada')
        r3 = ax.bar(x_pos + width - 0.3, CTEs[2], yerr=errors[2], width=width, align='center', alpha=0.5,
                    ecolor='black',
                    capsize=3, label='Precisão')
        r4 = ax.bar(x_pos + 2 * width - 0.3, CTEs[3], yerr=errors[3], width=width, align='center', alpha=0.5,
                    ecolor='black',
                    capsize=3, label='Sensitividade')
        r5 = ax.bar(x_pos + 3 * width - 0.3, CTEs[4], yerr=errors[4], width=width, align='center', alpha=0.5,
                    ecolor='black',
                    capsize=3, label='Especificidade')
        r6 = ax.bar(x_pos + 4 * width - 0.3, CTEs[5], yerr=errors[5], width=width, align='center', alpha=0.5,
                    ecolor='black',
                    capsize=3, label='F1-Score')

        ax.set_ylabel('%')
        ax.set_yticks(np.arange(0, 110, 10))
        ax.set_xticks(x_pos)
        ax.set_xticklabels(['KNN', 'SVM Linear', 'SVM RBF'])
        ax.set_title('Resultados da Classificação: ' + model_name)
        ax.yaxis.grid(True)
        ax.legend()

        autolabel(r1, ax)
        autolabel(r2, ax)
        autolabel(r3, ax)
        autolabel(r4, ax)
        autolabel(r5, ax)
        autolabel(r6, ax)

        # Save the figure and show
        plt.tight_layout()
        fig.savefig(outPathC + os.sep + 'plot' + os.sep + model_name + '_plot.png')
        fig.clear()


def process_data(model_name_list, process_time):
    for model_name in model_name_list:  # Iterate CNN models
        print('processing: ' + model_name + ' deep model')

        header_results = ['round', 'classifier', 'process_time']
        header_results[2:2] = metrics_name_list

        with open(outPathC + os.sep + 'metrics' + os.sep + model_name + '_results.csv', mode='a') as csv_file:
            csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            csv_writer.writerow(header_results)

        data = np.load(outPathF + os.sep + model_name + ".npy")

        for rnd in range(classification_rounds):  # Iterations of rounds

            print('\tprocessing: round ' + str(rnd + 1))

            np.random.shuffle(data)
            X_train, y_train, X_test, y_test = load_train_test(data, set_split['train_set'], set_split['test_set'],
                                                               classes)
            X_train, y_train = shuffle(X_train, y_train)
            X_test, y_test = shuffle(X_test, y_test)

            if ImagesID:
                id_test = X_test[:, 0].reshape(-1)
                X_test = X_test[:, 1:]
                X_train = X_train[:, 1:]

            X_train = normalize_x(X_train)
            X_test = normalize_x(X_test)

            with open(outPathC + os.sep + 'rounds' + os.sep + model_name + '_round' + str(rnd + 1) + '.csv',
                      'a') as csv_predict:
                csv_writer = csv.writer(csv_predict, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                csv_writer.writerow(['test_img' + str(i) for i in range(sum(set_split['test_set']))])
                if ImagesID:
                    csv_writer.writerow(['test_id', np.asarray(id_test)])
                csv_writer.writerow(['test_class', np.asarray(y_test)])

            for idx, (clf, clf_name, param_dist, n_iter_search) in enumerate(zip(clf_list, classifiers_name_list,
                                                                                 param_dist_list, n_iter_search_list)):
                # Start chronometer
                timeStart = current_milli_time()

                print('\t\tprocessing: ' + clf_name + ' classifier')

                if (param_dist != None):
                    random_search = RandomizedSearchCV(clf, param_distributions=param_dist, n_iter=n_iter_search,
                                                       n_jobs=-1, cv=cv)
                    random_search.fit(X_train, y_train)
                    clf = random_search.best_estimator_
                else:
                    clf.fit(X_train, y_train)

                predict = clf.predict(X_test)

                with open(outPathC + os.sep + 'rounds' + os.sep + model_name + '_round' + str(rnd + 1) + '.csv',
                          'a') as csv_predict:
                    csv_writer = csv.writer(csv_predict, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(['predict_' + clf_name, np.asarray(predict)])

                # Compute metrics
                accuracy = accuracy_score(y_test, predict)
                bal_accuracy = balanced_accuracy_score(y_test, predict)
                precision = precision_score(y_test, predict, average='macro')
                sensitivity = sensitivity_score(y_test, predict, average='macro')
                specificity = specificity_score(y_test, predict, average='macro')
                f1 = f1_score(y_test, predict, average='macro')

                # End chronometer
                process_time = current_milli_time() - timeStart

                # Generate confusion matrix and save per round
                cmx = confusion_matrix(y_test, predict)
                cmx_cols = classes
                cmx_rows = classes
                csv_matrix = pd.DataFrame(cmx, index=cmx_rows, columns=cmx_cols)
                csv_matrix.to_csv(outPathC + '/rounds/' + model_name + '_CMX' + str(rnd + 1) + '_' + clf_name + '.csv',
                                  sep=',')

                row = [rnd + 1, clf_name, accuracy * 100, bal_accuracy * 100, precision * 100, sensitivity * 100,
                       specificity * 100, f1 * 100, process_time]

                with open(outPathC + os.sep + 'metrics' + os.sep + model_name + '_results.csv', mode='a') as csv_file:
                    csv_writer = csv.writer(csv_file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
                    csv_writer.writerow(row)

        print('Load process: Complete', process_time)
        print("\n.......Loading: New Model.......\n")


def create_folder_structure():
    if not os.path.exists('./results/features'):
        os.makedirs('./results/features')
    if not os.path.exists('./results/classification/metrics'):
        os.makedirs('./results/classification/metrics')
    if not os.path.exists('./results/classification/plot'):
        os.makedirs('./results/classification/plot')
    if not os.path.exists('./results/classification/rounds'):
        os.makedirs('./results/classification/rounds')

def open_txt(path):
    lines = []
    file = open(path)
    raw = file.readlines()
    file.close()
    for line in raw:
        lines.append(line.split(' '))

    return lines


def create_images_from_txt():
    data = open_txt('./extraction/ocr_car_numbers_rotulado.txt')
    id = 0
    for line in data:
        digit = re.sub(r"[^a-zA-Z0-9 ]", "", line[len(line) - 1])
        if not os.path.exists('./images/ocr_car/' + digit):
            os.mkdir('./images/ocr_car/' + digit)

        image = [255 if line[i] == '1' else 0 for i in range(len(line) - 1)]
        new_img = Image.new("L", (35, 35), "white")
        new_img.putdata(image)
        new_img.save('./images/ocr_car/' + digit + "/" + str(id) + ".jpeg")
        id = id + 1


if __name__ == "__main__":
    #create_images_from_txt()

    ImagesID = False
    srcPath = './images/ocr_car'
    classes = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    class_weight = {0: 1, 1: 1, 2: 1, 3: 1, 4: 1, 5: 1, 6: 1, 7: 1, 8: 1, 9: 1}
    set_split = {'total_per_class': [
        308,
        397,
        313,
        348,
        413,
        326,
        363,
        274,
        313,
        297,
    ],  # Total of images per class
        'test_set': [
            308 * 20 // 100,
            397 * 20 // 100,
            313 * 20 // 100,
            348 * 20 // 100,
            413 * 20 // 100,
            326 * 20 // 100,
            363 * 20 // 100,
            274 * 20 // 100,
            313 * 20 // 100,
            297 * 20 // 100
        ],  # Images for test
        'train_set': [
            308 - (308 * 20 // 100),
            397 - (397 * 20 // 100),
            313 - (313 * 20 // 100),
            348 - (348 * 20 // 100),
            413 - (413 * 20 // 100),
            326 - (326 * 20 // 100),
            363 - (363 * 20 // 100),
            274 - (274 * 20 // 100),
            313 - (313 * 20 // 100),
            297 - (297 * 20 // 100),
        ]}
    model_name_list = ['ResNet50', 'MobileNet', 'DenseNet201']
    classifiers_name_list = ['KNN', 'SVM_Linear', 'SVM_RBF']
    metrics_name_list = ['accuracy', 'balanced_accuracy', 'precision', 'sensitivity', 'specificity', 'f1_score']
    classification_rounds = 10
    outPathF = './results/features'
    outPathC = './results/classification'
    create_folder_structure()

    ## For each deep model
    for model_name in model_name_list:
        process_time = deep_extractor(model_name, classes, ['csv'], srcPath, outPathF)
        print('Load process: Complete', process_time)

    print('\n\nDone')

    # Initialize classifiers
    clf_list = np.array([MultinomialNB(),
                         MLPClassifier(max_iter=1000, solver='adam', learning_rate_init=5e-04),
                         KNeighborsClassifier(),
                         RandomForestClassifier(class_weight=class_weight),
                         svm.SVC(kernel='linear', class_weight=class_weight, probability=True, max_iter=3000, tol=1e-3),
                         svm.SVC(kernel='poly', class_weight=class_weight, probability=True, max_iter=3000, tol=1e-3),
                         svm.SVC(kernel='rbf', class_weight=class_weight, probability=True, max_iter=3000, tol=1e-3)
                         ])

    # Specify parameters and distributions to classifiers
    param_dist_list = np.array([
        # KNN
        {"n_neighbors": [1, 3, 5, 7, 9, 11]},
        # SVM Linear
        {'kernel': ['linear'], 'C': [2 ** i for i in range(-5, 15)]},
        # SVM RBF
        {'kernel': ['rbf'], 'gamma': [2 ** i for i in range(-15, 3)],
         'C': [2 ** i for i in range(-5, 15)]}
    ])

    clf_numbers = {'Nearest_Neighbors': 2, 'SVM_Linear': 4, 'SVM_RBF': 6}

    idx_clf = np.array([clf_numbers[i] for i in classifiers_name_list])
    clf_list = clf_list[idx_clf]
    param_dist_list = param_dist_list[idx_clf]
    n_iter_search_list = np.array([0, 20, 5, 15, 20, 20, 20])
    n_iter_search_list = n_iter_search_list[idx_clf]
    cv = 3  # Random Search K-Fold

    process_data(model_name_list, process_time)
    plot_result(model_name_list, classifiers_name_list)

    print('\n\nDone')
