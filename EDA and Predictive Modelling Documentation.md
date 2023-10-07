# Exploratory Data Analysis and Predictive Modeling for Housing Prices

## Objective

The aim of this analysis is to predict the median value of owner-occupied homes ('MEDV') in Boston, based on various features like crime rate, average number of rooms per dwelling, pupil-teacher ratio, and more.

## Approach

### Data Exploration
- **Dataset**: The dataset contains 506 samples with 14 features.
- **Missing Values**: No missing values were identified.
- **Summary Statistics**: Features had different ranges, indicating the need for scaling.

### Feature Selection

1. **LASSO Regression**: LASSO helped in identifying features like 'RM', 'PTRATIO', and 'LSTAT' as the most important.

### Model Building and Evaluation

1. **Initial Models**: Multiple Linear Regression and Random Forest models were built using the common features. Random Forest outperformed Linear Regression with an \(R^2\) value of 0.782.
2. **Feature Update**:  Included 'PTRATIO' along with 'CRIM', improving the \(R^2\) to 0.842.
4. **Hyperparameter Tuning**: Used Grid Search to fine-tune the Random Forest model. The optimized model achieved an \(R^2\) of 0.840 on the test set.

## Conclusions and Recommendations

- **Important Features**: 'RM', 'LSTAT', 'CRIM', and PTRATIO were identified as significant predictors for the median value of homes.
- **Best Model**: The optimized Random Forest model with an \(R^2\) of 0.840 was the best performing model.
  



