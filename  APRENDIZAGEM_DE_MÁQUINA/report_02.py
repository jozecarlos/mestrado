from math import sqrt
from random import randrange

import numpy as np
import pandas as pd
from fitter import Fitter
from sklearn import preprocessing
from sklearn.impute import KNNImputer, SimpleImputer
from sklearn.metrics import mean_squared_error, mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier


def normal_dist(x, mean, sd):
    prob_density = (np.pi * sd) * np.exp(-0.5 * ((x - mean) / sd) ** 2)
    return prob_density


def generate_values_by_column(df, n):
    M = 0.3  # scale factor
    columns = df.columns.tolist()
    values = []
    for col in columns[0:-1]:
        # print(np.min(df[col].values),np.max(df[col].values))
        u1 = np.random.uniform(np.min(df[col].values), np.max(df[col].values), 1000)
        u2 = np.random.uniform(np.min(df[col].values), np.max(df[col].values), 1000)
        mean = np.mean(u1)
        sd = np.std(u1)

        # Apply function to the data.
        pdf = normal_dist(u1, mean, sd)
        # print(len(pdf))  ## só pra verificar se estão vindo 1000 numeros
        idx, = np.where(u2 <= pdf / M)  # rejection criterion
        # escolhe n valores pra adicionar ao dataset
        values.append(u1[idx][:n])

    return values


def item_a():
    distribution_by_column = []
    df = pd.read_csv('./datasets/output.csv')
    # Forma dos dados
    print(f'Number of rows: {df.shape[0]} | Columns (variables): {df.shape[1]}')

    # Verificando valores ausentes
    df.isna().sum()

    # Colunas
    columns = df.columns.tolist()
    classes = pd.unique(df["class"].values)

    # Escolhendo a melhor distribuição para cada atributo e treinando
    for col in columns[0:-1]:
        f = Fitter(df[col].values, distributions=['gamma', 'lognorm', "beta", "chi2", "norm"])
        f.fit()
        """
          melhores distribuições (em termos de ajuste). Uma vez feito o ajuste, pode-se querer obter 
          os parâmetros correspondentes à melhor distribuição. você precisará consultar a documentação do scipy 
          para descobrir quais são esses parâmetros (média, sigma, forma, ...). 
          Por conveniência, fornecemos o PDF correspondente:
        """
        print(col)
        """
          Vai imprimir a melhor distribuição para o atributo usando o square root error
        """
        print(f.get_best(method='sumsquare_error'))

    for c in classes:
        values = generate_values_by_column(df, 300)
        for i in range(0, 259):
            row = []
            for j in range(len(columns) - 1):
                row.append(values[j][i])
            row.append(c)
            df.loc[len(df.index)] = row

    # Forma dos dados
    print(f'Number of rows: {df.shape[0]} | Columns (variables): {df.shape[1]}')


def print_rmse(X_train, X_test, y_train, y_test, le):
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train, y_train)

    knn_test_mse = round(mean_squared_error(le.transform(y_test), le.transform(knn_model.predict(X_test))), 3)
    knn_test_mae = round(mean_absolute_error(le.transform(y_test), le.transform(knn_model.predict(X_test))), 3)
    knn_test_score = round(knn_model.score(X_test, y_test), 3)

    testing_ev = [['Model', 'MSE', 'MAE', 'RMSE', 'Prediction Score'],
                  ['KNN', knn_test_mse, knn_test_mae, round(sqrt(knn_test_mse), 3), knn_test_score]]

    s = pd.DataFrame(testing_ev[1:], columns=testing_ev[0])
    print("---------------------------")
    print(s.to_string(index=False))


def item_b():
    df = pd.read_csv('./datasets/iris_dataset.csv')
    columns = df.columns.tolist()
    le = preprocessing.LabelEncoder()
    # Forma dos dados
    print(f'Number of rows: {df.shape[0]} | Columns (variables): {df.shape[1]}')

    X = df.iloc[:, :-1].values
    y = df["target"].values
    le.fit(y)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    print_rmse(X_train, X_test, y_train, y_test, le)

    for _ in range(1, 151):
        row = randrange(2, 150)
        col = columns[randrange(0, 3)]
        df.at[row, col] = np.nan

    X2 = df.iloc[:, :-1].values

    imputer = KNNImputer(n_neighbors=5)
    imp_mean = SimpleImputer(missing_values=np.nan, strategy='mean')

    X3 = imputer.fit_transform(X2)
    X4 = imp_mean.fit_transform(X2)

    X_train2, X_test2, y_train2, y_test2 = train_test_split(X3, y, test_size=0.2, random_state=12)
    X_train3, X_test3, y_train3, y_test3 = train_test_split(X4, y, test_size=0.2, random_state=12)

    print_rmse(X_train2, X_test2, y_train2, y_test2, le)
    print_rmse(X_train3, X_test3, y_train3, y_test3, le)


if __name__ == "__main__":
    item_a()
    item_b()
