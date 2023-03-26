import pandas as pd
from pathlib2 import Path

best_results = pd.read_csv(str(Path.cwd().parent / 'all_models_result.csv'))

# For the different parameters, we want to see the best performing model
# We sort for every error and print the top (or bottom) line

errors = [ 'max_error', 'mean_absolute_percentage_error', 'mean_absolute_error',
           'mean_squared_error']
scores = ['explained_variance_score', 'r2_score']

# For errors, we want to see the model with the lowest error
for error in errors:
    best_results = best_results.sort_values(by=error)
    print(f"Best model for {error} is: \n")
    print(f" {best_results['model'].values[0]} model with data {best_results['data_option'].values[0]} and doing pca " 
                 f"{best_results['pca_option'].values[0]} with {best_results['n_components'].values[0]} components \n")



# For scores, we want to see the model with the highest score
for score in scores:
    best_results = best_results.sort_values(by=score, ascending=False)
    print(f"Best model for {score} is:")
    print(f" {best_results['model'].values[0]} model with data {best_results['data_option'].values[0]} and doing pca " 
                 f"{best_results['pca_option'].values[0]} with {best_results['n_components'].values[0]} components \n")

