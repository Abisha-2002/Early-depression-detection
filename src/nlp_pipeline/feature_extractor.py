from sklearn.feature_extraction.text import TfidfVectorizer

class NLPFeatureExtractor:
    """Extract features from code-mixed text."""
    
    def __init__(self, ngram_range=(3, 5), max_features=500):
        self.ngram_range = ngram_range
        self.max_features = max_features
        self.tfidf_vectorizer = None
    
    def extract_tfidf(self, texts):
        """Extract TF-IDF features from texts."""
        self.tfidf_vectorizer = TfidfVectorizer(
            analyzer='char',
            ngram_range=self.ngram_range,
            max_features=self.max_features
        )
        tfidf_matrix = self.tfidf_vectorizer.fit_transform(texts)
        return tfidf_matrix
    
    def extract_features(self, texts):
        """Extract all NLP features from texts."""
        return self.extract_tfidf(texts)