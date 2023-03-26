import pandas as pd

from sklearn.linear_model import LinearRegression, ElasticNet, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor

from data_handling import test_model
from plotting import plot_performance
def model_selection(x_train, y_train, x_test, y_test, model, data_option, pca_option, n_components, df_results):
    """
    Selects and trains the model
    :param x_train: train data set
    :param y_train: train target
    :param x_test: test data set
    :param y_test: test target
    :param model: type of model
    :param data_option: type of data scaling used
    :param pca_option: type of pca used
    :param n_components: number of components
    :param df_results: dataframe to store results
    :return: dataframe with results
    """

    if model == 'linearregression':
        lin_reg = LinearRegression()
        lin_reg.fit(x_train, y_train)

        results, y_pred = test_model(x_test, y_test, lin_reg)

    elif model == 'elasticnet':
        el_net = ElasticNet()
        el_net.fit(x_train, y_train)

        results, y_pred = test_model(x_test, y_test, el_net)
    elif model == 'lasso':
        lasso = Lasso()
        lasso.fit(x_train, y_train)

        results, y_pred = test_model(x_test, y_test, lasso)
    elif model == 'ridge':
        ridge = Ridge()
        ridge.fit(x_train, y_train)

        results, y_pred = test_model(x_test, y_test, ridge)
    elif model == 'randomforest':
        rf = RandomForestRegressor()
        rf.fit(x_train, y_train)

        results, y_pred = test_model(x_test, y_test, rf)
    else:
        raise ValueError('model not valid')

    # Plotting the results
    plot_performance(y_test, y_pred, model, data_option, pca_option, n_components)

    # Getting the results added to the df_results
    params = {'model': model, 'data_option': data_option, 'pca_option': pca_option, 'n_components': n_components}
    params.update(results)
    df_results = pd.concat([df_results, pd.DataFrame([params])], ignore_index=True)

    return df_results
