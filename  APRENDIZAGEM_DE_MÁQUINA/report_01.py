from math import sqrt

from numpy import where
from numpy import meshgrid
from numpy import arange
from numpy import hstack
import pandas as pd
import seaborn as sns
from fitter import Fitter
from matplotlib import pyplot as plt
from scipy.stats import beta
from scipy.stats import burr
from scipy.stats import gamma
from scipy.stats import lognorm
from scipy.stats import norm
from sklearn import preprocessing
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.metrics import mean_absolute_error, ConfusionMatrixDisplay
from sklearn.metrics import mean_squared_error
from sklearn.mixture import BayesianGaussianMixture
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsRegressor
from sklearn.preprocessing import StandardScaler
from datasets import load_banana, load_ripley, load_two_moon


def get_best_distribution(distribution):
    counter = 0
    num = distribution[0]

    for i in distribution:
        curr_frequency = distribution.count(i)
        if (curr_frequency > counter):
            counter = curr_frequency
            num = i

    return num


def find_best_distribution(df):
    distribution = []
    for col in df.columns:
        if col.isnumeric():
            f = Fitter(df[col].values,
                       distributions=['gamma',
                                      'lognorm',
                                      "beta",
                                      "burr",
                                      "norm"])
            f.fit()
            f.summary()

            ## Tentar escolher a melhor distribuição, nota-se que burr, gamma, norm, e beta empatam
            distribution.append(list(f.get_best(method='sumsquare_error').keys())[0])
            print(f.get_best(method='sumsquare_error'))
            plt.show()

    return get_best_distribution(distribution)


def plot_density_chart(df):
    for col in df.columns[:-1]:
        sns.kdeplot(df[col])
        plt.show()


def item_a():
    datasets = ['./datasets/diabetes.csv', './datasets/breast_cancer.csv']
    sns.set(style="darkgrid")
    model = {'beta': beta, 'gamma': gamma, 'burr': burr, 'lognorm': lognorm, 'norm': norm}

    for dataset in datasets:
        print(dataset)
        print("-----------------")
        df = pd.read_csv(dataset)
        columns = df.columns.tolist()
        d = preprocessing.normalize(df.iloc[:, :-1])
        scaled_df = pd.DataFrame(d, columns=columns[0:-1])
        scaled_df[columns[-1]] = df[columns[-1]]

        plot_density_chart(scaled_df)
        distribution = find_best_distribution(scaled_df)

        print("Melhor distribuição " + distribution)

        for col in columns:
            fd = model.get(distribution)
            print(col)
            print(fd.fit(scaled_df[col]))
            print("---------------------------")


def item_b():
    df = pd.read_csv('./datasets/output.csv')

    # Forma dos dados
    print(f'Number of rows: {df.shape[0]} | Columns (variables): {df.shape[1]}')

    # Vamos ver as possíveis melhores variáveis para modelar o quilate
    sns.pairplot(df)

    # Verificando valores ausentes
    df.isna().sum()

    # Normalizando
    columns = df.columns.tolist()
    d = preprocessing.normalize(df.iloc[:, :-1])
    normalized_df = pd.DataFrame(d, columns=columns[0:-1])
    normalized_df[columns[-1]] = df[columns[-1]]

    X = normalized_df.iloc[:, :-1].values
    y = normalized_df["class"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    clf = GaussianNB()
    clf.fit(X_train, y_train)

    # Plotando a Matrix de confusão
    ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
    plt.show()
    item_b_3()


# PARTE B. Modelo Ingênuo Iem 3
def item_b_3():
    datasets = [load_ripley(), load_banana(), load_two_moon()]

    for dataset in datasets:
        print("-------------------------------")
        print(dataset.name)
        # Forma dos dados
        print(f'Number of rows: {dataset.data.shape[0]} | Columns (variables): {dataset.data.shape[1] + 1}')

        X = preprocessing.normalize(dataset.data)
        y = dataset.target
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

        clf = GaussianNB()
        clf.fit(X_train, y_train)

        # Plotando a Matrix de confusão
        ConfusionMatrixDisplay.from_estimator(clf, X_test, y_test)
        plt.show()

        plot_decision_sruface(X, y, clf)


def plot_decision_sruface(X, y, model):
    # definir limites do domínio
    min1, max1 = X[:, 0].min() - 1, X[:, 0].max() + 1
    min2, max2 = X[:, 1].min() - 1, X[:, 1].max() + 1

    # defina a escala x e y
    x1grid = arange(min1, max1, 0.1)
    x2grid = arange(min2, max2, 0.1)

    # criar todas as linhas e linhas da grade
    xx, yy = meshgrid(x1grid, x2grid)

    # achate cada grade em um vetor
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((len(r1), 1)), r2.reshape((len(r2), 1))

    # vetores de pilha horizontal para criar entrada x1,x2 para o modelo
    grid = hstack((r1, r2))

    model.fit(X, y)

    # fazer previsões para a grade
    yhat = model.predict_proba(grid)

    # mantenha apenas as probabilidades para a classe 0
    yhat = yhat[:, 0]

    # remodelar as previsões de volta em uma grade
    zz = yhat.reshape(xx.shape)

    # plote a grade dos valores x, y e z como uma superfície
    c = plt.contourf(xx, yy, zz, cmap='RdBu')

    # adicione uma legenda, chamada de barra de cores
    plt.colorbar(c)

    # criar gráfico de dispersão para amostras de cada classe
    for class_value in range(2):
        # obter índices de linha para amostras com esta classe
        row_ix = where(y == class_value)
        # criar dispersão dessas amostras
        plt.scatter(X[row_ix, 0], X[row_ix, 1], cmap='Paired')
    # show the plot
    plt.show()


def item_c():
    df = pd.read_csv('./datasets/concrete_data.csv')
    # Forma dos dados
    print(f'Number of rows: {df.shape[0]} | Columns (variables): {df.shape[1]}')

    # Vamos ver as possíveis melhores variáveis para modelar o quilate
    sns.pairplot(df)

    # Verificando valores ausentes
    df.isna().sum()

    df_new = df.loc[df['age'] == 28]
    print(f'Number of rows: {df.shape[0]} | Columns (variables): {df.shape[1]}')

    X = df_new.drop(['age', 'concrete_compressive_strength'], axis=1).values
    y = df_new["concrete_compressive_strength"].values
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=12)

    # Standardization
    scaler = StandardScaler()
    train_scaled = scaler.fit_transform(X_train)
    test_scaled = scaler.transform(X_test)

    # Instance and fit
    knn_model = KNeighborsRegressor(n_neighbors=5)
    knn_model_inv = KNeighborsRegressor(n_neighbors=5, weights='distance')
    bayes_mix_model = BayesianGaussianMixture(n_components=8)

    knn_model.fit(train_scaled, y_train)
    knn_model_inv.fit(train_scaled, y_train)
    bayes_mix_model.fit(train_scaled, y_train)

    knn_test_mse = round(mean_squared_error(y_test, knn_model.predict(test_scaled)), 3)
    knn_test_mae = round(mean_absolute_error(y_test, knn_model.predict(test_scaled)), 3)
    knn_test_score = round(knn_model.score(test_scaled, y_test), 3)

    knn_test_mse_inv = round(mean_squared_error(y_test, knn_model_inv.predict(test_scaled)), 3)
    knn_test_mae_inv = round(mean_absolute_error(y_test, knn_model_inv.predict(test_scaled)), 3)
    knn_test_score_inv = round(knn_model_inv.score(test_scaled, y_test), 3)

    gnb_model_mse = round(mean_squared_error(y_test, bayes_mix_model.predict(test_scaled)), 3)
    gnb_model_mae = round(mean_absolute_error(y_test, bayes_mix_model.predict(test_scaled)), 3)
    gnb_model_score = round(bayes_mix_model.score(test_scaled, y_test), 3)

    for col in df.columns:
        f = Fitter(df[col].values, distributions=['gamma', 'lognorm', "beta", "burr", "norm"])
        f.fit()
        """
          melhores distribuições (em termos de ajuste). Uma vez feito o ajuste, pode-se querer obter 
          os parâmetros correspondentes à melhor distribuição. você precisará consultar a documentação do scipy 
          para descobrir quais são esses parâmetros (média, sigma, forma, ...). 
          Por conveniência, fornecemos o PDF correspondente:
        """
        f.summary()
        print(col)
        """
          Vai imprimir a melhor distribuição para o atributo usando o square root error
        """
        print(f.get_best(method='sumsquare_error'))

    plt.show()

    testing_ev = [['Model', 'MSE', 'MAE', 'RMSE', 'Prediction Score'],
                  ['KNN', knn_test_mse, knn_test_mae, round(sqrt(knn_test_mse), 3), knn_test_score],
                  ['KNN INV', knn_test_mse_inv, knn_test_mae_inv, round(sqrt(knn_test_mse_inv), 3), knn_test_score_inv],
                  ['Gaussian', gnb_model_mse, gnb_model_mae, round(sqrt(gnb_model_mse), 3), gnb_model_score]]

    s = pd.DataFrame(testing_ev[1:], columns=testing_ev[0])
    print(s.to_string(index=False))


if __name__ == "__main__":
    # item_a()
    item_b()
    # item_c()
