import numpy as np
import pandas as pd
import matplotlib.pyplot as plt #plot charts
import seaborn as sns #used for data visualization
import warnings #avoid warning flash
warnings.filterwarnings('ignore')

df = pd.read_csv("diabetes.csv")
# print(df.head())
print("The dimension of the dataframe: ", df.shape)
# Check there are non-null values data
print(df.info())
# The minimum values of Glucose, BloodPressure, SkinThickness, Insulin and BMI cannot be 0.
# The maximum value of Insulin is 846.000000.
print(df.describe())


# Data cleaning

# Drop duplicate value
df = df.drop_duplicates()
print(pd.isnull(df).sum())

print(df[df['BloodPressure'] == 0].shape[0])
print(df[df['Glucose'] == 0].shape[0])
print(df[df['SkinThickness'] == 0].shape[0])
print(df[df['Insulin'] == 0].shape[0])
print(df[df['BMI'] == 0].shape[0])

#replacing 0 values with median or mean of that column
df['Glucose'] = df['Glucose'].replace(0, df['Glucose'].mean())#normal distribution
df['BloodPressure'] = df['BloodPressure'].replace(0, df['BloodPressure'].mean())#normal distribution
df['SkinThickness'] = df['SkinThickness'].replace(0, df['SkinThickness'].median())#skewed distribution
df['Insulin'] = df['Insulin'].replace(0, df['Insulin'].median())#skewed distribution
df['BMI'] = df['BMI'].replace(0, df['BMI'].median())#skewed distribution


print(df.describe())

sns.countplot(y='Outcome', data=df)
#plt.show()
df.hist(bins=10, figsize=(10, 10))
#plt.show()

# Pearson's Correlation Coefficient
corrmat=df.corr()
plt.figure(figsize=(10, 10))
sns.heatmap(corrmat, annot=True)
#plt.show()

# If Pearson's Correlation Coefficient is approach to 1, it means that they are highly correlated.
df_selected = df.drop(['BloodPressure', 'Insulin', 'DiabetesPedigreeFunction'], axis='columns')

# Quantile Transformation
from sklearn.preprocessing import QuantileTransformer
x = df_selected
columns = ['Pregnancies', 'Glucose', 'SkinThickness', 'BMI', 'Age', 'Outcome']
quantile = QuantileTransformer()
X = quantile.fit_transform(x)
df_new = pd.DataFrame(X, columns=columns)
df_new.head()
print(df_new.head())

plt.figure(figsize=(16,12))
sns.set_style(style='whitegrid')
plt.subplot(3,3,1)
sns.boxplot(x=df_new['Glucose'], data=df_new)
plt.subplot(3,3,2)
sns.boxplot(x=df_new['BMI'], data=df_new)
plt.subplot(3,3,3)
sns.boxplot(x=df_new['Pregnancies'], data=df_new)
plt.subplot(3,3,4)
sns.boxplot(x=df_new['Age'], data=df_new)
plt.subplot(3,3,5)
sns.boxplot(x=df_new['SkinThickness'], data=df_new)
#plt.show()

target_name = 'Outcome'
# given predictions - training data
y = df_new[target_name]
# dropping the Outcome column and keeping all other columns as X
X = df_new.drop(target_name, axis=1)

# TRAIN TEST SPLIT
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0)

# K Nearest Neighbours
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.model_selection import GridSearchCV

#List Hyperparameters to tune
knn = KNeighborsClassifier()
n_neighbors = list(range(15, 25))
p = [1, 2]
weights = ['uniform', 'distance']
metric = ['euclidean', 'manhattan', 'minkowski']

#convert to dictionary
hyperparameters = dict(n_neighbors=n_neighbors, p=p, weights=weights, metric=metric)

#Making model
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=knn, param_grid=hyperparameters, n_jobs=-1, cv=cv, scoring='f1', error_score=0)
best_model_grid = grid_search.fit(X_train, y_train)
#Best Hyperparameters Value
print('Best gridsearch leaf_size:', best_model_grid.best_estimator_.get_params()['leaf_size'])
print('Best gridsearch p:', best_model_grid.best_estimator_.get_params()['p'])
print('Best gridsearch n_neighbors:', best_model_grid.best_estimator_.get_params()['n_neighbors'])

#Predict testing set
knn_pred_grid = best_model_grid.predict(X_test)
print("Classification Report is:\n", classification_report(y_test, knn_pred_grid))
print("\n F1:\n", f1_score(y_test,knn_pred_grid))
print("\n Precision score is:\n", precision_score(y_test, knn_pred_grid))
print("\n Recall score is:\n", recall_score(y_test, knn_pred_grid))

# Bayes optimization
from bayes_opt import BayesianOptimization
from sklearn.model_selection import cross_val_score

# Function to optimize
def optimize_knn(n_neighbors, p, leaf_size):
    # Converting to integer because Bayesian Optimization will produce float values
    n_neighbors = int(n_neighbors)
    p = int(p)
    leaf_size = int(leaf_size)
    # Initialize the KNN model
    knn = KNeighborsClassifier(n_neighbors=n_neighbors, p=p, leaf_size=leaf_size)
    # Perform cross-validation
    scores = cross_val_score(knn, X_train, y_train, cv=cv, scoring='f1')
    # Return the mean F1 score
    return scores.mean()

# Set up Bayesian optimization
optimizer = BayesianOptimization(
    f=optimize_knn,
    pbounds={"n_neighbors": (15, 25), "p": (1, 2), "leaf_size": (20, 40)},
    random_state=1,
)
# Run optimization
optimizer.maximize(init_points=10, n_iter=25)

# Extract best hyperparameters
best_params_bayes = optimizer.max['params']
best_n_neighbors = int(best_params_bayes['n_neighbors'])
best_p = int(best_params_bayes['p'])
best_leaf_size = int(best_params_bayes['leaf_size'])

print('Best bayes n_neighbors:', best_n_neighbors)
print('Best bayes p:', best_p)
print('Best bayes leaf_size:', best_leaf_size)

# Initialize the KNN model with the best hyperparameters
best_knn = KNeighborsClassifier(n_neighbors=best_n_neighbors, p=best_p, leaf_size=best_leaf_size)
# Train the model on your training data
best_knn.fit(X_train, y_train)
# Make predictions on your test data
knn_pred_bayes = best_knn.predict(X_test)

print("Classification Report is:\n", classification_report(y_test, knn_pred_bayes))
print("\n F1:\n", f1_score(y_test, knn_pred_bayes))
print("\n Precision score is:\n", precision_score(y_test, knn_pred_bayes))
print("\n Recall score is:\n", recall_score(y_test, knn_pred_bayes))

# Random search
from sklearn.model_selection import RandomizedSearchCV
random_search = RandomizedSearchCV(estimator=knn, param_distributions=hyperparameters, n_jobs=-1, cv=cv, scoring='f1', error_score=0)
best_model_random = grid_search.fit(X_train, y_train)
#Best Hyperparameters Value
print('Best randomsearch leaf_size:', best_model_random.best_estimator_.get_params()['leaf_size'])
print('Best randomsearch p:', best_model_random.best_estimator_.get_params()['p'])
print('Best randomsearch n_neighbors:', best_model_random.best_estimator_.get_params()['n_neighbors'])

# Naive Bayes
from sklearn.naive_bayes import GaussianNB

param_grid_nb = {
    'var_smoothing': np.logspace(0, -2, num=100)
}

nbModel_grid = GridSearchCV(estimator=GaussianNB(), param_grid=param_grid_nb, verbose=1, cv=10, n_jobs=-1)
best_model = nbModel_grid.fit(X_train, y_train)
nb_pred = best_model.predict(X_test)
print("Classification Report is:\n", classification_report(y_test, nb_pred))
print("\n F1:\n", f1_score(y_test, nb_pred))
print("\n Precision score is:\n", precision_score(y_test, nb_pred))
print("\n Recall score is:\n", recall_score(y_test, nb_pred))

# Naive Bayes with Bayesian Optimization
# Function to optimize for Naive Bayes
def optimize_nb(var_smoothing):
    # Initialize the Naive Bayes model
    nb = GaussianNB(var_smoothing=var_smoothing)
    # Perform cross-validation
    scores = cross_val_score(nb, X_train, y_train, cv=cv, scoring='f1')
    # Return the mean F1 score
    return scores.mean()

# Set up Bayesian optimization for Naive Bayes
optimizer_nb = BayesianOptimization(
    f=optimize_nb,
    pbounds={"var_smoothing": (1e-9, 1e-1)},
    random_state=1,
)

# Run optimization for Naive Bayes
optimizer_nb.maximize(init_points=10, n_iter=25)

# Extract best hyperparameters for Naive Bayes
best_params_nb = optimizer_nb.max['params']
best_var_smoothing = best_params_nb['var_smoothing']

print('Best var_smoothing for Naive Bayes:', best_var_smoothing)

# Initialize the Naive Bayes model with the best hyperparameters
best_nb = GaussianNB(var_smoothing=best_var_smoothing)
# Train the model on your training data
best_nb.fit(X_train, y_train)
# Make predictions on your test data
nb_pred_bayes = best_nb.predict(X_test)

print("Classification Report for Naive Bayes is:\n", classification_report(y_test, nb_pred_bayes))
print("\n F1 for Naive Bayes:\n", f1_score(y_test, nb_pred_bayes))
print("\n Precision score for Naive Bayes is:\n", precision_score(y_test, nb_pred_bayes))
print("\n Recall score for Naive Bayes is:\n", recall_score(y_test, nb_pred_bayes))

# Support Vector Machine
from sklearn.svm import SVC

kernel = ['poly', 'rbf', 'sigmoid']
C = [50, 10, 1.0, 0.1, 0.01]
gamma = ['scale']

# define grid search
grid = dict(kernel=kernel, C=C, gamma=gamma)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=SVC(), param_grid=grid, n_jobs=-1, cv=cv, scoring='f1', error_score=0)

grid_result = grid_search.fit(X, y)
svm_pred_grid = grid_result.predict(X_test)
print("Classification Report is:\n", classification_report(y_test, svm_pred_grid))
print("\n F1:\n", f1_score(y_test, svm_pred_grid))
print("\n Precision score is:\n", precision_score(y_test, svm_pred_grid))
print("\n Recall score is:\n", recall_score(y_test, svm_pred_grid))

# bayes optimization
# Define hyperparameter ranges
param_ranges_svm = {
    "C": (0.01, 50),
    "kernel": (0, 2),  # Numeric representation for kernel choice
    "gamma": (0, 0)  # Numeric representation for gamma choice
}

# Function to optimize SVM
def optimize_svm(C, kernel, gamma):
    # Map numeric representation back to actual values
    kernel_choices = ['poly', 'rbf', 'sigmoid']
    gamma_choices = ['scale']
    kernel = kernel_choices[int(kernel)]
    gamma = gamma_choices[int(gamma)]

    # Initialize the SVM model
    svm = SVC(C=C, kernel=kernel, gamma=gamma)
    # Perform cross-validation
    scores = cross_val_score(svm, X_train, y_train, cv=cv, scoring='f1')

    # Return the mean F1 score
    return scores.mean()

# Set up Bayesian optimization for SVM
optimizer_svm = BayesianOptimization(
    f=optimize_svm,
    pbounds=param_ranges_svm,
    random_state=1,
)

# Run optimization for SVM
optimizer_svm.maximize(init_points=10, n_iter=25)

# Extract best hyperparameters for SVM
best_params_svm = optimizer_svm.max['params']
best_C_svm = best_params_svm['C']
best_kernel_svm = ['poly', 'rbf', 'sigmoid'][int(best_params_svm['kernel'])]
best_gamma_svm = 'scale'

print('Best C for Support Vector Machine:', best_C_svm)
print('Best kernel for Support Vector Machine:', best_kernel_svm)
print('Best gamma for Support Vector Machine:', best_gamma_svm)

# Initialize the Support Vector Machine model with the best hyperparameters
best_svm = SVC(C=best_C_svm, kernel=best_kernel_svm, gamma=best_gamma_svm)
# Train the model on your training data
best_svm.fit(X_train, y_train)
# Make predictions on your test data
svm_pred_bayes = best_svm.predict(X_test)

print("Classification Report for Support Vector Machine is:\n", classification_report(y_test, svm_pred_bayes))
print("\n F1 for Support Vector Machine:\n", f1_score(y_test, svm_pred_bayes))
print("\n Precision score for Support Vector Machine is:\n", precision_score(y_test, svm_pred_bayes))
print("\n Recall score for Support Vector Machine is:\n", recall_score(y_test, svm_pred_bayes))

# Decision Tree
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier(random_state=42)

# Create the parameter grid based on the results of random search
params = {
    'max_depth': [5, 10, 20, 25],
    'min_samples_leaf': [10, 20, 50, 100, 120],
    'criterion': ["gini", "entropy"]
}
grid_search = GridSearchCV(estimator=dt,
                           param_grid=params,
                           cv=4, n_jobs=-1, verbose=1, scoring="accuracy")

best_dt_grid = grid_search.fit(X_train, y_train)
dt_pred_grid = best_dt_grid.predict(X_test)
print("Classification Report is:\n", classification_report(y_test, dt_pred_grid))
print("\n F1:\n", f1_score(y_test, dt_pred_grid))
print("\n Precision score is:\n", precision_score(y_test, dt_pred_grid))
print("\n Recall score is:\n", recall_score(y_test, dt_pred_grid))


# Random Forest
from sklearn.ensemble import RandomForestClassifier

# define models and parameters
model = RandomForestClassifier()
n_estimators = [1800]
max_features = ['sqrt', 'log2']
# define grid search
grid = dict(n_estimators=n_estimators, max_features=max_features)
cv = RepeatedStratifiedKFold(n_splits=10, n_repeats=3, random_state=1)
grid_search = GridSearchCV(estimator=model, param_grid=grid, n_jobs=-1, cv=cv, scoring='accuracy', error_score=0)
best_rf_grid = grid_search.fit(X_train, y_train)
rf_pred_grid = best_rf_grid.predict(X_test)

print("Classification Report is:\n", classification_report(y_test, rf_pred_grid))
print("\n F1:\n", f1_score(y_test, rf_pred_grid))
print("\n Precision score is:\n", precision_score(y_test, rf_pred_grid))
print("\n Recall score is:\n", recall_score(y_test, rf_pred_grid))


# Logistic Regression


