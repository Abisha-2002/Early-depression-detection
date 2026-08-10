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
        sinhala_count = 0
        tamil_count = 0
        english_count = 0
        
        for char in token:
            # Sinhala Unicode range: 0x0D80 - 0x0DFF
            if 0x0D80 <= ord(char) <= 0x0DFF:
                sinhala_count += 1
            # Tamil Unicode range: 0x0B80 - 0x0BFF
            elif 0x0B80 <= ord(char) <= 0x0BFF:
                tamil_count += 1
            # English/ASCII characters
            elif ord(char) < 128:
                english_count += 1
        
        # Determine language based on majority
        if sinhala_count > tamil_count and sinhala_count > english_count:
            return 'sinhala'
        elif tamil_count > sinhala_count and tamil_count > english_count:
            return 'tamil'
        elif english_count > sinhala_count and english_count > tamil_count:
            return 'english'
        elif sinhala_count > 0:
            return 'sinhala'
        elif tamil_count > 0:
            return 'tamil'
        else:
            return 'english'
