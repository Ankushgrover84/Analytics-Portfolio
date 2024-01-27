# Importing necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso
from sklearn.model_selection import GridSearchCV
import numpy as np

# Load the dataset
df = pd.read_csv('housing.csv', header=None, delim_whitespace=True)
column_names = ['CRIM', 'ZN', 'INDUS', 'CHAS', 'NOX', 'RM', 'AGE', 'DIS', 'RAD', 'TAX', 'PTRATIO', 'B', 'LSTAT', 'MEDV']
df.columns = column_names
'''
Input features in order:
1) CRIM: per capita crime rate by town
2) ZN: proportion of residential land zoned for lots over 25,000 sq.ft.
3) INDUS: proportion of non-retail business acres per town
4) CHAS: Charles River dummy variable (1 if tract bounds river; 0 otherwise)
5) NOX: nitric oxides concentration (parts per 10 million) [parts/10M]
6) RM: average number of rooms per dwelling
7) AGE: proportion of owner-occupied units built prior to 1940
8) DIS: weighted distances to five Boston employment centres
9) RAD: index of accessibility to radial highways
10) TAX: full-value property-tax rate per $10,000 [$/10k]
11) PTRATIO: pupil-teacher ratio by town
12) B: The result of the equation B=1000(Bk - 0.63)^2 where Bk is the proportion of blacks by town
13) LSTAT: % lower status of the population

Output variable:
1) MEDV: Median value of owner-occupied homes in $1000's [k$]
'''
# Basic Statistical Summaries
print(df.describe())

# Basic Information
print(df.info())
# Setting up the aesthetics for the plots
sns.set(style="whitegrid")

# Initialize the figure
plt.figure(figsize=(20, 15))

# Create a list of feature names
features = df.columns

# Create subplots for each feature to check for outliers using boxplots
for i, feature in enumerate(features, 1):
    plt.subplot(4, 4, i)
    sns.boxplot(y=df[feature])
    plt.title(f'Boxplot of {feature}')

plt.tight_layout()
plt.show()

# Calculate the IQR for MEDV
Q1_MEDV = df['MEDV'].quantile(0.25)
Q3_MEDV = df['MEDV'].quantile(0.75)
IQR_MEDV = Q3_MEDV - Q1_MEDV

# Define bounds for the outliers
lower_bound_MEDV = Q1_MEDV - 1.5 * IQR_MEDV
upper_bound_MEDV = Q3_MEDV + 1.5 * IQR_MEDV

# Identify the outliers
outliers_MEDV = df[(df['MEDV'] < lower_bound_MEDV) | (df['MEDV'] > upper_bound_MEDV)]

# Display the outliers for MEDV
print(outliers_MEDV[['MEDV']])

# Remove rows where MEDV is equal to 50.0 as outliers
df_filtered = df[df['MEDV'] != 50.0]

# Calculate the correlation matrix for the filtered DataFrame
correlation_matrix = df_filtered.corr()

# Correlation Heatmap
high_correlation = correlation_matrix[(correlation_matrix > 0.74) | (correlation_matrix < -0.74)]
sns.heatmap(high_correlation, annot=True, fmt=".2f", cmap='coolwarm')
plt.title('High Correlation Matrix (Above 0.74 or Below -0.74)')
plt.show()
# Scatter Plots
sns.scatterplot(x='RM', y='MEDV', data=df)
plt.show()
sns.scatterplot(x='LSTAT', y='MEDV', data=df)
#plt.show()


# K-means Clustering
scaler = StandardScaler()
X_scaled = scaler.fit_transform(df_filtered)
# Determine the optimal number of clusters using the Elbow Method
inertia = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=0)
    kmeans.fit(X_scaled)
    inertia.append(kmeans.inertia_)

# Plotting the Elbow Method graph
plt.figure(figsize=(10, 6))
plt.plot(range(1, 11), inertia, marker='o')
plt.title('Elbow Method for Optimal Number of Clusters')
plt.xlabel('Number of Clusters')
plt.ylabel('Inertia')
plt.show()

# elbow method shows 3 clusters
kmeans = KMeans(n_clusters=3, random_state=42)
kmeans.fit(X_scaled)
labels = kmeans.labels_
df_clustered = df_filtered.copy()
df_clustered['Cluster'] = labels
cluster_summary = df_clustered.groupby('Cluster').mean()

print(cluster_summary)


# Linear Regression Model
features = ['RM', 'LSTAT','CRIM','PTRATIO']
X = df[features]
y = df['MEDV']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Scaling the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Applying LASSO
lasso = Lasso(alpha=1.0)
lasso.fit(X_train_scaled, y_train)

# Feature importance
feature_importance = lasso.coef_

# Pairing feature names with their importance scores
feature_importance_dict = dict(zip(X.columns, feature_importance))
sorted_features = sorted(feature_importance_dict.items(), key=lambda x: np.abs(x[1]), reverse=True)

print(sorted_features)


model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(r2)

# Extracting the coefficients of the common features in the model
coefficients_common = model.coef_

# Pairing feature names with their coefficients for interpretation
coefficients_dict = dict(zip(features, coefficients_common))
print(coefficients_dict)

##{'RM': 3.9654458420336525, 'LSTAT': -0.5177893841693868, 'PTRATIO': -0.940943586325589}

# Random Forest Model
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)
y_rf_pred = rf_model.predict(X_test)
rf_mse = mean_squared_error(y_test, y_rf_pred)
rf_r2 = r2_score(y_test, y_rf_pred)

print(rf_r2)

# Extracting updated feature importances
feature_importance_rf_updated = rf_model.feature_importances_
feature_importance_rf_updated_dict = dict(zip(features, feature_importance_rf_updated))
sorted_features_rf_updated = sorted(feature_importance_rf_updated_dict.items(), key=lambda x: x[1], reverse=True)

print(sorted_features_rf_updated)


'''
RM (Average number of rooms per dwelling): 
The most influential feature, explaining nearly 52% of the variance in 'MEDV'.

LSTAT (Lower status of the population): 
Explaining approximately 35% of the variance:

CRIM (Per capita crime rate by town): 
The feature 'CRIM' explains about 10% of the variance

PTRATIO (Pupil-teacher ratio by town):
 The pupil-teacher ratio has the least impact among the three features, explaining approximately 3% of the variance in 'MEDV'

 R2 of linear regression is 0.62os and with Random Forest the R2 is 0.85
 '''

# Data for plotting
features = [item[0] for item in sorted_features_rf_updated]
importances = [item[1] for item in sorted_features_rf_updated]

# Create a horizontal bar chart for feature importances
plt.figure(figsize=(10, 6))
plt.barh(features, importances, color='skyblue')
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance as Determined by Random Forest')
plt.gca().invert_yaxis()  # Reverse the order to have the most important feature at the top
plt.show()

# Define a simplified hyperparameter grid to search
simplified_param_grid = {
    'n_estimators': [50, 100],  # Reduced number of trees
    'max_features': ['auto', 'sqrt'],  # Reduced feature options
    'max_depth': [None, 10],  # Reduced depth options
    'min_samples_split': [2, 5],  # Reduced split options
    'min_samples_leaf': [1, 2]  # Reduced leaf options
}

# Create the simplified Grid Search model
simplified_grid_search = GridSearchCV(estimator=rf_model, param_grid=simplified_param_grid, 
                                      cv=3, n_jobs=-1, verbose=2, scoring='neg_mean_squared_error')

# Fit the simplified Grid Search to the data
simplified_grid_search.fit(X_train, y_train)

# Get the best parameters and best MSE from the simplified Grid Search
best_params_simplified = simplified_grid_search.best_params_
best_mse_simplified = -simplified_grid_search.best_score_

print(best_params_simplified)
print(best_mse_simplified)
