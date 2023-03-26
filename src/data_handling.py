import pandas as pd
from pathlib2 import Path
from sklearn.metrics import explained_variance_score, max_error, mean_absolute_percentage_error, mean_absolute_error, mean_squared_error,  r2_score

def get_data(data_option):
    """
    get the data set from file system
    :param data_option: 'no_scaled', 'minmax', 'robust'
    :return: train, test data set
    """
    path_to_folder = Path.cwd().parent / 'data'
    if data_option == 'no_scaled':

        train = pd.read_csv(str(path_to_folder / 'train.csv'))
        test = pd.read_csv(str(path_to_folder / 'test.csv'))
    elif data_option == 'minmax':
        train = pd.read_csv(str(path_to_folder / 'train_minmax.csv'))
        test = pd.read_csv(str(path_to_folder / 'test_minmax.csv'))
    elif data_option == 'robust':
        train = pd.read_csv(str(path_to_folder / 'train_robust.csv'))
        test = pd.read_csv(str(path_to_folder / 'test_robust.csv'))
    else:
        raise ValueError('data_option not valid')
    return train, test

def test_model(x_test, y_test, fitted_model):
    """
    Tests the performance of the model
    :param x_test: test data set
    :param y_test: test target
    :param fitted_model: fitted model
    :return: dict with errors and predictions
    """

    # Predict and determine errors
    y_pred = fitted_model.predict(x_test)
    errors = [explained_variance_score(y_test, y_pred), max_error(y_test, y_pred), mean_absolute_percentage_error(y_test, y_pred),
              mean_absolute_error(y_test, y_pred), mean_squared_error(y_test, y_pred), r2_score(y_test, y_pred)]
    # round errors to only 3 decimals
    errors = [round(error, 3) for error in errors]
    columns = ['explained_variance_score', 'max_error', 'mean_absolute_percentage_error', 'mean_absolute_error', 'mean_squared_error', 'r2_score']
    dict_results = dict(zip(columns, errors))

    return dict_results, y_pred