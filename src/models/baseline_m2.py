import numpy as np
from transformers import AutoTokenizer, AutoModel
import torch
from sklearn.metrics import f1_score, accuracy_score

class BaselineM2:
    """M2: Off-the-shelf multilingual BERT (mBERT), no fine-tuning."""
    
    def __init__(self, model_name='bert-base-multilingual-cased'):
        self.model_name = model_name
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)
        self.model.eval()
    
    def get_embeddings(self, texts):
        """Get BERT embeddings for texts."""
        embeddings = []
        for text in texts:
            inputs = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
            with torch.no_grad():
                outputs = self.model(**inputs)
                embedding = outputs.last_hidden_state.mean(dim=1).squeeze().numpy()
                embeddings.append(embedding)
        return np.array(embeddings)
    
    def train(self, X_train, y_train):
        """Placeholder - mBERT is used off-the-shelf."""
        print("M2: Using off-the-shelf mBERT embeddings")
        return self
    
    def predict(self, X_test):
        """Placeholder - would need a classifier."""
        print("M2: Prediction requires a trained classifier on top of embeddings")
        return np.zeros(len(X_test))
    
    def evaluate(self, X_test, y_test):
        """Evaluate model performance."""
        predictions = self.predict(X_test)
        return {
            'accuracy': accuracy_score(y_test, predictions),
            'f1_macro': f1_score(y_test, predictions, average='macro')
        }

if __name__ == "__main__":
    print("M2: Off-the-shelf mBERT Baseline")