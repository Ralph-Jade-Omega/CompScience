import numpy as np

class NaiveBayes:
    def __init__(self):
        self.class_priors = {}
        self.feature_likelihoods = {}

    def fit(self, X, y):
        # Calculate P(Class)
        self.classes = np.unique(y)
        for cls in self.classes:
            self.class_priors[cls] = np.sum(y == cls) / len(y)

            # Calculate P(Feature|Class)
            self.feature_likelihoods[cls] = {}
            for feature_index in range(X.shape[1]):
                feature_values = X[:, feature_index]
                self.feature_likelihoods[cls][feature_index] = {}
                for value in np.unique(feature_values):
                    self.feature_likelihoods[cls][feature_index][value] = np.sum((feature_values == value) & (y == cls)) / np.sum(y == cls)

    def predict(self, X):
        predictions = []
        for x in X:
            class_probabilities = {}
            for cls in self.classes:
                prior = self.class_priors[cls]
                likelihood = 1
                for feature_index in range(X.shape[1]):
                    likelihood *= self.feature_likelihoods[cls][feature_index].get(x[feature_index], 0)
                class_probabilities[cls] = prior * likelihood
            predictions.append(max(class_probabilities, key=class_probabilities.get))
        return np.array(predictions)

# Example dataset for playing golf
if __name__ == "__main__":
    # Features: Outlook, Temperature, Humidity, Windy
    # Labels: Play Golf (Yes/No)
    X = np.array([['Sunny', 'Hot', 'High', 'False'],
                  ['Sunny', 'Hot', 'High', 'True'],
                  ['Overcast', 'Hot', 'High', 'False'],
                  ['Rainy', 'Mild', 'High', 'False'],
                  ['Rainy', 'Cool', 'Normal', 'False'],
                  ['Rainy', 'Cool', 'Normal', 'True'],
                  ['Overcast', 'Cool', 'Normal', 'True'],
                  ['Sunny', 'Mild', 'High', 'False'],
                  ['Sunny', 'Cool', 'Normal', 'False'],
                  ['Rainy', 'Mild', 'Normal', 'False'],
                  ['Sunny', 'Mild', 'Normal', 'True'],
                  ['Overcast', 'Mild', 'High', 'True'],
                  ['Overcast', 'Hot', 'Normal', 'False'],
                  ['Rainy', 'Mild', 'High', 'True']])
    
    y = np.array(['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes', 'Yes', 'No'])

    # Create and train the model
    model = NaiveBayes()
    model.fit(X, y)

    # Test input: Outlook = Rainy, Temperature = Mild, Humidity = High, Windy = False
    test_data = np.array([['Rainy', 'Mild', 'High', 'False']])
    
    # Make prediction
    prediction = model.predict(test_data)
    print("Prediction for playing golf:", prediction[0])