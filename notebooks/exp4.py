# Import necessary libraries for data handling, modeling, and tracking
import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import mlflow
from mlflow.models import infer_signature
import dagshub


dagshub.init(repo_owner="franztac", repo_name="Water--End-to-End-ML_Proj", mlflow=True)
mlflow.set_experiment("Experiment 4")
mlflow.set_tracking_uri("https://dagshub.com/franztac/Water--End-to-End-ML_Proj.mlflow")


# Load and preprocess data
data = pd.read_csv("C:/Users/UTENTE/Git Repos/water_potability.csv")
train_data, test_data = train_test_split(data, test_size=0.20, random_state=42)


# Function to fill missing values with mean
def fill_missing_with_mean(df_original):
    df = df_original.copy()
    for column in df.columns:
        if df[column].isnull().any():
            mean_value = df[column].mean()
            df[column] = df[column].fillna(mean_value)
    return df


# Fill missing values with mean for training and test sets
train_processed_data = fill_missing_with_mean(train_data)
test_processed_data = fill_missing_with_mean(test_data)


# Prepare the training data by separating features and target variable
X_train = train_processed_data.drop(columns=["Potability"], axis=1)  # Features
y_train = train_processed_data["Potability"]  # Target variable


# Define the Random Forest Classifier model and the parameter distribution for hyperparameter tuning
rf = RandomForestClassifier(random_state=42)
param_dist = {
    "n_estimators": [100, 200, 300, 500, 1000],
    "max_depth": [None, 4, 5, 6, 10],
}

# Perform RandomizedSearchCV to find the best hyperparameters for the Random Forest model
random_search = RandomizedSearchCV(
    estimator=rf,
    param_distributions=param_dist,
    n_iter=50,
    cv=5,
    n_jobs=-1,
    verbose=2,
    random_state=42,
)


# Start a new MLflow run to log the Random Forest tuning process
with mlflow.start_run(run_name="Random Forest Tuning") as parent_run:
    # Fit the RandomizedSearchCV object on the training data to identify the best hyperparameters
    random_search.fit(X_train, y_train)

    # Log the parameters and mean test scores for each combination tried
    for i in range(len(random_search.cv_results_["params"])):
        with mlflow.start_run(
            run_name=f"Combination {i + 1}", nested=True
        ) as child_run:
            mlflow.log_params(
                random_search.cv_results_["params"][i]
            )  # Log the parameters
            mlflow.log_metric(
                "mean_test_score", random_search.cv_results_["mean_test_score"][i]
            )  # Log the mean test score

    # Print the best hyperparameters found by RandomizedSearchCV
    print("Best parameters found: ", random_search.best_params_)

    # Log the best parameters in MLflow
    mlflow.log_params(random_search.best_params_)

    # Train the model using the best parameters identified by RandomizedSearchCV
    best_rf = random_search.best_estimator_
    best_rf.fit(X_train, y_train)

    # Save the trained model to a file for later use
    pickle.dump(best_rf, open("model.pkl", "wb"))

    # Prepare test data
    X_test = test_processed_data.iloc[:, 0:-1].values
    y_test = test_processed_data.iloc[:, -1].values

    # Load the saved model from the file
    model = pickle.load(open("model.pkl", "rb"))

    # Make predictions on the test set using the loaded model
    y_pred = model.predict(X_test)

    # Calculate and print performance metrics: accuracy, precision, recall, and F1-score
    acc = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1_score = f1_score(y_test, y_pred)

    # Log performance metrics into MLflow for tracking
    mlflow.log_metric("acc", acc)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1-score", f1_score)

    # Let's log dataset as well
    train_df = mlflow.data.from_pandas(train_processed_data)
    test_df = mlflow.data.from_pandas(test_processed_data)
    mlflow.log_input(train_df, "train")
    mlflow.log_input(test_df, "test")

    # Log the current script file as an artifact in MLflow
    mlflow.log_artifact(__file__)

    # Infer the model signature using the test features and predictions
    sign = infer_signature(X_test, random_search.best_estimator_.predict(X_test))

    # Log the trained model in MLflow with its signature
    mlflow.sklearn.log_model(
        random_search.best_estimator_, "Best Model", signature=sign
    )

    # Print the calculated performance metrics to the console for review
    print("acc", acc)
    print("precision", precision)
    print("recall", recall)
    print("f1-score", f1_score)
