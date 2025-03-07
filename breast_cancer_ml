import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class BreastCancerPrediction:
    def __init__(self, n_estimators=100, max_depth=None):
        self.n_estimators = n_estimators
        self.max_depth = max_depth
        self.model = RandomForestClassifier(n_estimators=self.n_estimators, max_depth=self.max_depth, random_state=42)

    # Load the data and save it in a dataframe
    def load_data(self):
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        self.x = df.drop(columns=["target"])
        self.y = df["target"]

    # Split data into Train/Test data
    def split_data(self, test_size=0.2):
        self.x_train, self.x_test, self.y_train, self.y_test = train_test_split(self.x, self.y, test_size=test_size, random_state=42)

    # Hyperparameter Tuning and Cross-Validation
    def tune_and_cross_validate(self):
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20]
        }

        # GridSearchCV does Cross-Validation
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(self.x_train, self.y_train)

        # Best model from GridSearch
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        # Cross-Validation of the best models
        cv_scores = cross_val_score(self.best_model, self.x_train, self.y_train, cv=5)
        return cv_scores

    # Visualize importance of features
    def plot_feature_importance(self):
        importances = self.best_model.feature_importances_
        indices = importances.argsort()[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [self.x.columns[i] for i in indices])
        plt.xlabel("Feature Importance")
        plt.show()

    # evaluate the model
    def test_best_model(self):
        best_accuracy = accuracy_score(self.y_test, self.best_model.predict(self.x_test))
        return best_accuracy

    # Summarized output
    def summary(self):
        print(f"Best Random Forest Parameters: {self.best_params}")
        print(f"Best Random Forest Test Accuracy: {self.test_best_model():.4f}")
        self.plot_feature_importance()

# Main function
def main():
    cancer_predictor = BreastCancerPrediction()
    cancer_predictor.load_data()
    cancer_predictor.split_data()

    # Tuning and Cross-Validation
    cv_scores = cancer_predictor.tune_and_cross_validate()
    print(f"Cross Validation Scores: {cv_scores}")

    # summarize output
    cancer_predictor.summary()

if __name__ == "__main__":
    main()
