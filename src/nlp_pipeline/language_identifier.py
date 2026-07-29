import re

class LanguageIdentifier:
    """Identify language at token level for code-mixed text."""
    
    def __init__(self):
        self.model = None
    
    def identify_tokens(self, text):
        """Identify language for each token in text."""
        if not text or not isinstance(text, str):
            return []
        
        tokens = re.findall(r'\S+', text)
        token_languages = []
        
        for token in tokens:
            lang = self._identify_single_token(token)
            token_languages.append((token, lang))
        
        return token_languages
    
    def _identify_single_token(self, token):
        """Identify language of a single token."""
        for char in token:
            if 0x0D80 <= ord(char) <= 0x0DFF:
                return 'sinhala'
            if 0x0B80 <= ord(char) <= 0x0BFF:
                return 'tamil'
        return 'english'