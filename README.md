# Predicting prices of houses
In this project we have a dataset with transactional data from houses in Beijing. 
The goal is to predict the price of a house for transactions in 2017 and onwards, 
based on the transactions before that period. Data description is in the task.pdf file.

### Quick Start
1. requirements.txt has all the required packages.
    - compiled from requirements.in using pip-tools
    - Written using Python 3.11.2
    - All the code can be run using a virtualenv and installing the requirements.txt
2. Place the `Dataset.csv` in the data folder.
3. Order of operations:
   1. Run exploration.ipynb <-- pre processes the data
   2. Run visualization.ipynb <-- visualization, more pre-processing and some feature engineering
   3. Run src/main.py <-- trains the model and makes predictions; takes a while
   4. run src/best_results_picker.py <-- picks the best results of all the trained models
4. figures are saved in the figures folder and `all_moddels_results.csv` is saved in same folder as this readme. That file contains all the results.

## Data Exploration (and a bit of engineering)
This readme is a summary of the work done in the exploration notebook. For more thorough walkthrough, please check the notebook.
- The data has 24 columns and 318850 rows.
- There's all sorts of issues: missing values, outliers, wrong data types, strings in the columns, etc.
- There's 32 bad rows we removed.
- NaN values in most columns are removed (the columns with low prevalence of NaNs).
- For one category (`floorPosition`) the NaN values are used to create a new category.
- One column (DOM) has almost half missing values. This column was imputed with the median value.

## Data engineering.
- `square` and `ladderRatio` had some extreme outliers, these were removed.
- one-hot encoding was done for all the categories with more than 2 categories in them.
- Decision was made to delete the tradeTime column, because we use it for splitting the data and how it dips right after 2017 <-- might've been a bad decision
- After all this the shape of the data was 297838 rows with 43 columns.
- We saw that plotting the coordinates on a map with the price as a heatmap that there's a correlate with higher 
- prices towards the center of Beijing. <-- we didn't specifically/directly exploit this observation. 
- We explored different scaling methods but decided to use them all and see which one works best.

## Model selection, training and parameter tuning.
This problem is a regression problem. We can try several regression models and see which one works best. 
An addition to those regression models are RandomForests.
We'll use the following models:
- LinearRegression
- Ridge
- Lasso
- ElasticNet
- RandomForestRegressor
These models were trained using normal python files with some parameter tuning in what scaling 
to use `'no_scaled', 'minmax', 'robust'` and if doing pca how many components to use `[3, 5, 7, 9, 11]`.

## Results.
1. The best models are the RandomForests. Those are also the most computationally intensive models.
2. The linear models 
3. Random forest are better without PCA
4. A lot of models have a negative R2 scores. Suggesting it's worse than just a straight linear line. Could strongly suggest overfitting on the training data.
5. Some non-pca linear regressions still have low mean_absolute errors on the test data. 

## Considerations for improvement.
Exploring more of the results and the models that came out of it can give a lot more insights. 
Likely they'll answer some questions below too.
1. To the RandomForests itself no parameter tuning was used. This could have fine-tuned the results even better.
2. More importantly, RandomForests can work fine without doing the one-hot encoding. 
3. We could have dealt with a lot less features if we hadn't done that for the randomForests.
4. We could have reduced more of the features, a lot had very low correlation. We did this with the PCA to some extent. But we could already have thrown out the lowest 20 correlated ones.
5. That, with other ways to address overfitting, could have made the models better.
6. We could re-transform the scaled data back to the original scale. Since the scaling affects some errors.
7. A lot of the models seem to under-predict the actual price. The dateTime column was removed to try and avoid over predicting, 
   considering how the data set was split up. Re-introducing it might make models better. Especially since it does correlate quite a bit with price. 
8. Didn't get the kernel PCA to work because it required too much memory.
9. Also didn't exploit the location dependency towards city center directly.
10. We can add figures and zooms of figures to illustrate our point better in this readme.
