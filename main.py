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

    # Laden der Daten
    def load_data(self):
        data = load_breast_cancer()
        df = pd.DataFrame(data.data, columns=data.feature_names)
        df['target'] = data.target
        self.X = df.drop(columns=["target"])
        self.y = df["target"]

    # Aufteilen der Daten
    def split_data(self, test_size=0.2):
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=test_size, random_state=42)

    # Hyperparameter Tuning und Cross-Validation
    def tune_and_cross_validate(self):
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [None, 10, 20]
        }

        # GridSearchCV führt auch Cross-Validation durch
        grid_search = GridSearchCV(estimator=self.model, param_grid=param_grid, cv=5, n_jobs=-1)
        grid_search.fit(self.X_train, self.y_train)

        # Bestes Modell aus dem GridSearch
        self.best_model = grid_search.best_estimator_
        self.best_params = grid_search.best_params_

        # Cross-Validation der besten Modelle
        cv_scores = cross_val_score(self.best_model, self.X_train, self.y_train, cv=5)
        return cv_scores

    # Feature-Importances visualisieren
    def plot_feature_importance(self):
        importances = self.best_model.feature_importances_
        indices = importances.argsort()[::-1]

        plt.figure(figsize=(10, 6))
        plt.title("Feature Importances")
        plt.barh(range(len(indices)), importances[indices], align="center")
        plt.yticks(range(len(indices)), [self.X.columns[i] for i in indices])
        plt.xlabel("Feature Importance")
        plt.show()

    # Modell evaluieren
    def test_best_model(self):
        best_accuracy = accuracy_score(self.y_test, self.best_model.predict(self.X_test))
        return best_accuracy

    # Zusammenfassende Ausgaben
    def summary(self):
        print(f"Best Random Forest Parameters: {self.best_params}")
        print(f"Best Random Forest Test Accuracy: {self.test_best_model():.4f}")
        self.plot_feature_importance()

# Hauptfunktion
def main():
    cancer_predictor = BreastCancerPrediction()
    cancer_predictor.load_data()
    cancer_predictor.split_data()

    # Tuning und Cross-Validation
    cv_scores = cancer_predictor.tune_and_cross_validate()
    print(f"Cross Validation Scores: {cv_scores}")

    # Zusammenfassende Ausgaben
    cancer_predictor.summary()

if __name__ == "__main__":
    main()