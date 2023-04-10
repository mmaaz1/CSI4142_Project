import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('https://raw.githubusercontent.com/mmaaz1/CSI4142_Project/main/Datasets/covid_vs_emissions.csv')

# Select the features
features = ['Total_cases', 'Total_deaths', 'Population', 'Area(sq._km.)', 'Net_migration_rate', 'Gdp_per_capita', 
            'Literacy_rate', 'Unemployment_rate']
data.dropna(subset = features, inplace = True)
X = data[features]

# Standardize the data
scaler = StandardScaler()

# Fit to data, then transform it.
X_scaled = scaler.fit_transform(X)

# Instantiate the One-Class SVM model
outlierProportion = 0.05

# Unsupervised Outlier Detection is used in our case.
# Gamma can be set to 'scale'.
models = OneClassSVM(kernel = "rbf", nu = outlierProportion, gamma = 0.1)

# Fit the model
# Compute the mean and std to be used for later scaling.
models.fit(X_scaled)

# Predict the outliers
predictedOutlier = models.predict(X_scaled)

# Mark the outliers in the dataset
# Add the outlier predictions to the original dataset.
data['Outlier'] = predictedOutlier

# Filter and Display the outliers
# One-class SVM returns either 1 or -1 for each data. Data with -1 is
# considered as an outlier so we need to filer the ones with the value of -1
outliers = data[data['Outlier'] == -1]
print("Outliers Detected:", len(outliers))
print("\n Outliers:")
print(outliers)
print("\nthe end")
print("\n")
