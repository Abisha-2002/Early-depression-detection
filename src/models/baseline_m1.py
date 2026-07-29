from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, accuracy_score

class BaselineM1:
    """M1: TF-IDF + Logistic Regression (Local literature baseline)."""
    
    def __init__(self, random_state=42):
        self.tfidf = TfidfVectorizer(max_features=1000)
        self.model = LogisticRegression(
            multi_class='multinomial',
            max_iter=1000,
            random_state=random_state
        )
    
    def train(self, X_train, y_train):
        """Train the baseline model."""
        X_train_tfidf = self.tfidf.fit_transform(X_train)
        self.model.fit(X_train_tfidf, y_train)
        return self
    
    def predict(self, X_test):
        """Make predictions."""
        X_test_tfidf = self.tfidf.transform(X_test)
        return self.model.predict(X_test_tfidf)
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'f1_macro': f1_score(y_test, predictions, average='macro')
        }

if __name__ == "__main__":
    print("M1: TF-IDF + Logistic Regression Baseline")