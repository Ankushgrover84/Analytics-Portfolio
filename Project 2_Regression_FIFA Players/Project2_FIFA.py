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
from sklearn.preprocessing import PolynomialFeatures
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV

# Load the dataset
df = pd.read_csv('Project2_FIFAplayers_22.csv')
# Drop url and other columns
extra_columns = ['sofifa_id', 'player_url', 'long_name', 'dob', 'club_loaned_from',
                   'nation_position', 'nation_jersey_number', 'body_type', 'real_face',
                   'player_face_url', 'club_logo_url', 'nation_logo_url', 'nation_flag_url',
                    'goalkeeping_speed', 'player_tags', 'nation_team_id','club_flag_url']

df = df.drop(extra_columns, axis = 1)

# Basic Information
print(df.info())

#check for nulls
print(df.isnull().sum())

# Drop rows where column 'value_eur' has null values
df = df.dropna(subset=['value_eur'])
print(df.head())

# Generate summary statistics for numerical columns
numerical_summary = df.describe()

print(numerical_summary)

# Generate summary statistics for categorical columns
categorical_summary = df.describe(include=['object'])

print(categorical_summary)

# Plot the distribution of 'overall' player ratings
plt.figure(figsize=(10, 6))
sns.histplot(df['overall'], bins=30, color='blue', kde=True)
plt.title('Distribution of Overall Player Ratings')
plt.xlabel('Overall Rating')
plt.ylabel('Frequency')
plt.show()

# Plot the distribution of player ages
plt.figure(figsize=(10, 6))
sns.histplot(df['age'], bins=30, color='orange', kde=True)
plt.title('Distribution of Player Ages')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# Calculate the correlation matrix for key numerical variables
correlation_matrix = df[['overall', 'potential', 'value_eur', 'wage_eur', 'age']].corr()

# Plot the correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap of Key Numerical Variables')
plt.show()

# Create a boxplot to visualize the distribution of player wages across different leagues
plt.figure(figsize=(18, 10))
sns.boxplot(data=df, x='league_name', y='value_eur')
plt.xticks(rotation=90)
plt.title('Player value_eur by League')
plt.xlabel('League Name')
plt.ylabel('value_eur in EUR')
plt.yscale('log')  # Using log scale for better visibility
plt.show()

# List of columns that represent player skills in different positions
player_skill_columns = ['rw', 'cf', 'st', 'lw', 'cam', 'cm', 'lm', 'cdm', 'cb', 'lb', 'rb', 'rm', 'lwb', 'ldm', 'rdm', 'rwb', 'ls', 'rs']

# Extract these columns for analysis
df_player_skills = df[player_skill_columns]

# Show basic statistics to understand these columns better
print(df_player_skills.describe())

# Define a function to convert the skill ratings to numerical format
def convert_skill_rating(skill):
    try:
        if '+' in skill:
            base, added = skill.split('+')
            return int(base) + int(added)
        elif '-' in skill:
            base, sub = skill.split('-')
            return int(base) - int(sub)
        else:
            return int(skill)
    except:
        return skill

# Apply the conversion function to each of the skill columns
for col in player_skill_columns:
    df[col] = df[col].apply(convert_skill_rating)

# Drop rows with NaN values in skill-related columns
df = df.dropna(subset=player_skill_columns)

# Show basic statistics again to confirm conversion
print(df[player_skill_columns].describe())

# Define a dictionary with the current column names and their more descriptive versions
rename_dict = {
    'rw': 'RightWinger',
    'cf': 'CenterForward',
    'st': 'Striker',
    'lw': 'LeftWinger',
    'cam': 'CentralAttackingMidfielder',
    'cm': 'CentralMidfielder',
    'lm': 'LeftMidfielder',
    'cdm': 'CentralDefensiveMidfielder',
    'cb': 'CenterBack',
    'lb': 'LeftBack',
    'rb': 'RightBack',
    'rm': 'RightMidfielder',
    'lwb': 'LeftWingBack',
    'ldm': 'LeftDefensiveMidfielder',
    'rdm': 'RightDefensiveMidfielder',
    'rwb': 'RightWingBack',
    'ls': 'LeftStriker',
    'rs': 'RightStriker'
}

# Rename the columns
df.rename(columns=rename_dict, inplace=True)

# Define additional skill-related columns for further analysis
additional_skill_columns = [
    'pace', 'shooting', 'passing', 'dribbling', 'defending', 'physic',
    'attacking_crossing', 'attacking_finishing', 'attacking_heading_accuracy',
    'attacking_short_passing', 'attacking_volleys',
    'skill_dribbling', 'skill_curve', 'skill_fk_accuracy', 'skill_long_passing', 'skill_ball_control',
    'movement_acceleration', 'movement_sprint_speed', 'movement_agility', 'movement_reactions', 'movement_balance',
    'power_shot_power', 'power_jumping', 'power_stamina', 'power_strength', 'power_long_shots',
    'mentality_aggression', 'mentality_interceptions', 'mentality_positioning', 'mentality_vision', 'mentality_penalties', 'mentality_composure',
    'defending_marking_awareness', 'defending_standing_tackle', 'defending_sliding_tackle',
    'goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes'
]

# Combine these with the renamed position-specific skill columns
all_skill_columns = list(rename_dict.values()) + additional_skill_columns

# Show basic statistics for these additional skill-related columns
print(df[additional_skill_columns].describe())

# Goalkeepers are indicated by 'GK' in the 'player_positions' column. 

# Create a new DataFrame for goalkeepers based on the 'player_positions' column
df_goalkeepers = df[df['player_positions'].str.contains('GK')]

# Create another DataFrame for outfield players
df_outfield_players = df[~df['player_positions'].str.contains('GK')]

# Drop rows with NaN values in skill-related columns
df_outfield_players_cleaned = df_outfield_players.dropna(subset=all_skill_columns)

# Prepare the data again
X_outfield_cleaned = df_outfield_players_cleaned[all_skill_columns]
y_outfield_cleaned = df_outfield_players_cleaned['value_eur']

# Split the data into training and test sets
X_train_outfield_cleaned, X_test_outfield_cleaned, y_train_outfield_cleaned, y_test_outfield_cleaned = train_test_split(X_outfield_cleaned, y_outfield_cleaned, test_size=0.2, random_state=42)

# Initialize and fit the Linear Regression model
lr_model_outfield_cleaned = LinearRegression()
lr_model_outfield_cleaned.fit(X_train_outfield_cleaned, y_train_outfield_cleaned)

# Make predictions
y_pred_outfield_cleaned = lr_model_outfield_cleaned.predict(X_test_outfield_cleaned)

# Calculate metrics
lr_rmse_outfield_cleaned = np.sqrt(mean_squared_error(y_test_outfield_cleaned, y_pred_outfield_cleaned))
lr_r2_outfield_cleaned = r2_score(y_test_outfield_cleaned, y_pred_outfield_cleaned)

print(lr_rmse_outfield_cleaned)
print(lr_r2_outfield_cleaned)

## R2 = 0.40, RMSE = $6,472,680

# Calculate the correlation matrix for the X fields (skill-related columns)
correlation_matrix = X_outfield_cleaned.corr()

# Generate a heatmap for the correlation matrix
plt.figure(figsize=(20, 20))
sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', square=True)
plt.title('Correlation Matrix of Skill-Related Columns for Outfield Players')
plt.show()

# To remove one of each pair of highly correlated fields, we'll set a threshold for high correlation
correlation_threshold = 0.9

# Create an empty set to hold correlated variables
correlated_variables = set()

# Iterate over the correlation matrix
for i in range(len(correlation_matrix.columns)):
    for j in range(i):
        if abs(correlation_matrix.iloc[i, j]) > correlation_threshold:
            colname = correlation_matrix.columns[i]
            correlated_variables.add(colname)

# Remove these correlated columns from the DataFrame
X_outfield_reduced = X_outfield_cleaned.drop(columns=correlated_variables)

# Show the columns that have been removed and those that remain
correlated_variables, X_outfield_reduced.columns.tolist()

# Define the columns related to goalkeeping skills
goalkeeping_columns = ['goalkeeping_diving', 'goalkeeping_handling', 'goalkeeping_kicking', 'goalkeeping_positioning', 'goalkeeping_reflexes']

# Remove these columns from the reduced set of features
X_outfield_reduced_no_gk = X_outfield_reduced.drop(columns=goalkeeping_columns)

# Split the data into training and test sets using this further reduced set of features
X_train_outfield_reduced_no_gk, X_test_outfield_reduced_no_gk, y_train_outfield_reduced_no_gk, y_test_outfield_reduced_no_gk = train_test_split(X_outfield_reduced_no_gk, y_outfield_cleaned, test_size=0.2, random_state=42)

# Initialize and fit the Linear Regression model
lr_model_outfield_reduced_no_gk = LinearRegression()
lr_model_outfield_reduced_no_gk.fit(X_train_outfield_reduced_no_gk, y_train_outfield_reduced_no_gk)

# Make predictions
y_pred_outfield_reduced_no_gk = lr_model_outfield_reduced_no_gk.predict(X_test_outfield_reduced_no_gk)

# Calculate metrics
lr_rmse_outfield_reduced_no_gk = np.sqrt(mean_squared_error(y_test_outfield_reduced_no_gk, y_pred_outfield_reduced_no_gk))
lr_r2_outfield_reduced_no_gk = r2_score(y_test_outfield_reduced_no_gk, y_pred_outfield_reduced_no_gk)

print(lr_rmse_outfield_reduced_no_gk)
print(lr_r2_outfield_reduced_no_gk)

## R2 reduced to 0.29, RMSE increased to 7,029,857

# Initialize and fit the LassoCV model
# LassoCV performs cross-validation to find the best alpha (regularization parameter)
lasso_model_outfield = LassoCV(cv=5, random_state=42)
lasso_model_outfield.fit(X_train_outfield_reduced_no_gk, y_train_outfield_reduced_no_gk)

# Make predictions
y_pred_outfield_lasso = lasso_model_outfield.predict(X_test_outfield_reduced_no_gk)

# Calculate metrics
lasso_rmse_outfield = np.sqrt(mean_squared_error(y_test_outfield_reduced_no_gk, y_pred_outfield_lasso))
lasso_r2_outfield = r2_score(y_test_outfield_reduced_no_gk, y_pred_outfield_lasso)

# Best alpha value chosen by LassoCV
best_alpha = lasso_model_outfield.alpha_

print(lasso_rmse_outfield)
print(lasso_r2_outfield)
print(best_alpha)

## Similar R2 as before

# Use LassoCV to select important features
sfm = SelectFromModel(lasso_model_outfield, threshold=1e-5)
sfm.fit(X_train_outfield_reduced_no_gk, y_train_outfield_reduced_no_gk)

# Transform the data to keep only the most important features
X_train_outfield_important = sfm.transform(X_train_outfield_reduced_no_gk)
X_test_outfield_important = sfm.transform(X_test_outfield_reduced_no_gk)

# Initialize and fit a new Linear Regression model using only important features
lr_model_outfield_important = LinearRegression()
lr_model_outfield_important.fit(X_train_outfield_important, y_train_outfield_reduced_no_gk)

# Make predictions
y_pred_outfield_important = lr_model_outfield_important.predict(X_test_outfield_important)

# Calculate metrics
lr_rmse_outfield_important = np.sqrt(mean_squared_error(y_test_outfield_reduced_no_gk, y_pred_outfield_important))
lr_r2_outfield_important = r2_score(y_test_outfield_reduced_no_gk, y_pred_outfield_important)

# Identify the important features
important_features = X_train_outfield_reduced_no_gk.columns[sfm.get_support()]

print(lr_rmse_outfield_important)
print(lr_r2_outfield_important) 
print(important_features.tolist())

#R2 with important terms didn't improve

# Perform some feature engineering to create new features that may better represent the problem
# For this example, let's create some interaction terms among the important features

# Initialize the PolynomialFeatures transformer with degree=2 to create interaction terms
poly = PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)

# Fit and transform the data using only the important features
X_train_outfield_poly = poly.fit_transform(X_train_outfield_reduced_no_gk[important_features])
X_test_outfield_poly = poly.transform(X_test_outfield_reduced_no_gk[important_features])

# Initialize and fit a new Linear Regression model using the engineered features
lr_model_outfield_poly = LinearRegression()
lr_model_outfield_poly.fit(X_train_outfield_poly, y_train_outfield_reduced_no_gk)

# Make predictions
y_pred_outfield_poly = lr_model_outfield_poly.predict(X_test_outfield_poly)

# Calculate metrics
lr_rmse_outfield_poly = np.sqrt(mean_squared_error(y_test_outfield_reduced_no_gk, y_pred_outfield_poly))
lr_r2_outfield_poly = r2_score(y_test_outfield_reduced_no_gk, y_pred_outfield_poly)

print(lr_rmse_outfield_poly)
print(lr_r2_outfield_poly)

# Extract coefficients
coefficients = lr_model_outfield_poly.coef_

# Create a DataFrame for feature importances
feature_importance_df = pd.DataFrame({
    'Feature': important_features,  
    'Importance': coefficients
})

# Sort the DataFrame by the absolute value of the Importance
feature_importance_df['Abs_Importance'] = feature_importance_df['Importance'].abs()
feature_importance_df = feature_importance_df.sort_values('Abs_Importance', ascending=False)

# Plot
import matplotlib.pyplot as plt
plt.figure(figsize=(10, 6))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Feature Importance from Linear Regression')
plt.show()






'''
#Root Mean Square Error (RMSE): improved 5,020,334 EUR
#R² improved: 0.642

# Initialize and fit a Random Forest Regressor to use for feature selection
rf_model = RandomForestRegressor(n_estimators=75, max_depth=15, max_features='auto', n_jobs=-1, random_state=42)

rf_model.fit(X_train_outfield_poly, y_train_outfield_reduced_no_gk)

# Use SelectFromModel to select important features based on the Random Forest model
sfm_rf = SelectFromModel(rf_model, threshold=1e-3)  # threshold is set low to capture more features initially
sfm_rf.fit(X_train_outfield_poly, y_train_outfield_reduced_no_gk)

# Transform the data to keep only the most important features
X_train_outfield_poly_important_rf = sfm_rf.transform(X_train_outfield_poly)
X_test_outfield_poly_important_rf = sfm_rf.transform(X_test_outfield_poly)

# Initialize and fit a new Linear Regression model using only important features
lr_model_outfield_poly_important_rf = LinearRegression()
lr_model_outfield_poly_important_rf.fit(X_train_outfield_poly_important_rf, y_train_outfield_reduced_no_gk)

# Make predictions
y_pred_outfield_poly_important_rf = lr_model_outfield_poly_important_rf.predict(X_test_outfield_poly_important_rf)

# Calculate metrics
lr_rmse_outfield_poly_important_rf = np.sqrt(mean_squared_error(y_test_outfield_reduced_no_gk, y_pred_outfield_poly_important_rf))
lr_r2_outfield_poly_important_rf = r2_score(y_test_outfield_reduced_no_gk, y_pred_outfield_poly_important_rf)

# Number of important features selected by Random Forest
num_important_features_rf = X_train_outfield_poly_important_rf.shape[1]

print(lr_rmse_outfield_poly_important_rf)
print(lr_r2_outfield_poly_important_rf) 
print(num_important_features_rf)

#RMSE increased 5134617.951882978
#R2 reduced 0.6256505691148242


# Get feature importances from the Random Forest model
feature_importances = rf_model.feature_importances_

# Create a DataFrame for the feature importances
feature_importance_df = pd.DataFrame({
    'Feature': X_train_outfield_poly_important_rf.columns,
    'Importance': feature_importances
})

# Sort the DataFrame by the importances
feature_importance_df = feature_importance_df.sort_values(by='Importance', ascending=False)

# Plot the top 10 most important features
plt.figure(figsize=(12, 8))
plt.barh(feature_importance_df['Feature'][:10], feature_importance_df['Importance'][:10])
plt.xlabel('Importance')
plt.ylabel('Feature')
plt.title('Top 10 Most Important Features in Random Forest Model')
plt.show()

# Return the top 10 most important features
#feature_importance_df.head(10)
'''