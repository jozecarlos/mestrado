import pandas as pd
import pkg_resources
from sklearn.preprocessing import LabelEncoder
from sklearn.utils import Bunch
from sklearn.utils.validation import check_X_y


def __file_path_of(s):
    myfile_path = pkg_resources.resource_filename('.', s)
    print(myfile_path)
    return myfile_path


def __encode_and_transform(data):
    X, y = data.iloc[:, :-1], data.iloc[:, -1]
    X, y = check_X_y(X, y, multi_output=True, y_numeric=True)

    le = LabelEncoder()
    bunch = Bunch(data=X, target=le.fit_transform(y), name=data.name)

    return bunch


def load_vertebral_column_2c():
    return load_vertebral_column('datasets/column_2C.dat')


def load_vertebral_column_3c():
    return load_vertebral_column('datasets/column_3C.dat')


def load_vertebral_column(filename):
    data = pd.read_csv(filename, sep=' ', header=None)
    data.name = 'Vertebral Coluna'
    return __encode_and_transform(data)


def load_banana():
    data = pd.read_csv('datasets/datasets-7627-10826-banana.csv', sep=',')
    data.name = 'Banana'
    return __encode_and_transform(data)

def load_german_stalog():
    data = pd.read_csv('datasets/german.data-numeric', sep='  ')
    data.name = 'German'
    return __encode_and_transform(data)

# https://archive.ics.uci.edu/ml/datasets/Haberman%27s+Survival
def load_habermans_survivor():
    data = pd.read_csv('datasets/haberman.data', sep=',')
    data.name = 'Haberman'
    return __encode_and_transform(data)

# https://archive.ics.uci.edu/ml/datasets/Heart+Disease
def load_heart_disease():

    return None

# https://archive.ics.uci.edu/ml/datasets/Ionosphere
def load_ionosphere():
    data = pd.read_csv('datasets/ion.csv', sep=',')
    return __encode_and_transform(data)


def load_two_moon():
    from sklearn.datasets import make_moons
    X, y = make_moons(n_samples=300)

    return Bunch(data=X, target=y, name='Two_moom')


def load_ripley():
    data = pd.read_csv('datasets/rip.csv', sep=',', header=None)
    data.name = 'Ripley'
    return __encode_and_transform(data)

