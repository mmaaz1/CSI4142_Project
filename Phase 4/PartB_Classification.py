import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, precision_score, recall_score
from sklearn.preprocessing import KBinsDiscretizer
import time

# Load the data
data = pd.read_csv('https://raw.githubusercontent.com/mmaaz1/CSI4142_Project/main/Datasets/covid_vs_emissions.csv')

# Select the features
features = ['Total_cases', 'Total_deaths', 'Population', 'Area(sq._km.)', 'Net_migration_rate', 'Gdp_per_capita', 
            'Literacy_rate', 'Unemployment_rate']
data.dropna(subset=features, inplace=True)
X = data[features]

# Initialize model results
accuracy = {"dt":0, "rf":0, "gb":0}
precision = {"dt":0, "rf":0, "gb":0}
recall = {"dt":0, "rf":0, "gb":0}
times = {"dt":0, "rf":0, "gb":0}

# Predict values of all three emissions
targets = ['Co2_change_n', 'Co4_change_n', 'N2o_change_n']
for target in targets:
    # Use binning to convert output classes from normalized to categorical values
    y = data[target]
    discretizer = KBinsDiscretizer(n_bins=10, strategy='uniform', encode='ordinal')
    y_discrete = discretizer.fit_transform(y.values.reshape(-1, 1)).ravel()

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y_discrete, test_size=0.2, random_state=42)

    # Instantiate the three models
    models = {
        "dt":DecisionTreeRegressor(random_state=42),
        "rf":RandomForestRegressor(random_state=42),
        "gb":GradientBoostingRegressor(random_state=42)
    }

    # Train, Predict and Evaluate the models
    for name, model in models.items():
        start_time = time.time()
        model.fit(X_train, y_train)
        times[name] += time.time() - start_time

        preds = model.predict(X_test)

        accuracy[name] += accuracy_score(y_test, preds.round())
        precision[name] += precision_score(y_test, preds.round(), average='macro', zero_division=1)
        recall[name] += recall_score(y_test, preds.round(), average='macro', zero_division=1)

# Print the results for each model
models = {
    "Decision Tree":"dt",
    "Gradient Boosting":"gb",
    "Random Forest":"rf"
}
for name, model in models.items():
    print(name)
    print("Accuracy: {:.2f}".format(accuracy[model]/3))
    print("Precision: {:.2f}".format(precision[model]/3))
    print("Recall: {:.2f}".format(recall[model]/3))
    print("Execution Time: {:.2f} seconds".format(times[model]/3))
    print("\n")
