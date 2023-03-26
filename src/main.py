"""

"""
import pandas as pd
from pathlib2 import Path

from data_handling import get_data
from pca import reduce_components
from models import model_selection


data_options = ['no_scaled', 'minmax', 'robust']
pca_options = ['no_pca', 'pca']
list_n_components = [3, 5, 7, 9, 11]
models = ['linearregression', 'elasticnet', 'lasso', 'ridge', 'randomforest']

df_results = pd.DataFrame(columns=['model', 'data_option', 'pca_option', 'n_components', 'explained_variance_score', 'max_error',
                                   'mean_absolute_percentage_error', 'mean_absolute_error', 'mean_squared_error',  'r2_score'])
for data_option in data_options:
    train, test = get_data(data_option)
    for pca_option in pca_options:
        no_pca_count = 0
        for n_components in list_n_components:
            if pca_option == 'no_pca':
                no_pca_count += 1
                n_components = None
                if no_pca_count > 1:
                    break
            x_train, x_test = reduce_components(train, test, pca_option, n_components)
            for model in models:
                print(f"Modelling {model} with data {data_option} and {pca_option} with {n_components}")
                df_results = model_selection(x_train, train['price'], x_test, test['price'], model, data_option, pca_option, n_components, df_results)
                df_results.to_csv(str(Path.cwd().parent / 'all_models_result.csv'))








