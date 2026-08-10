import unicodedata
import re

class UnicodeNormalizer:
    """Normalise Unicode characters in Sinhala/Tamil/English text."""
    
    @staticmethod
    def normalize(text):
        """Normalise Unicode text to standard form."""
        if not text or not isinstance(text, str):
            return text
        
        # Normalize Unicode (NFC form)
        text = unicodedata.normalize('NFC', text)
        
        # Remove control characters
        text = re.sub(r'[\x00-\x1f\x7f-\x9f]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    @staticmethod
    def is_sinhala_char(char):
        """Check if character is Sinhala."""
        return 0x0D80 <= ord(char) <= 0x0DFF
    
    @staticmethod
    def is_tamil_char(char):
        """Check if character is Tamil."""
        return 0x0B80 <= ord(char) <= 0x0BFF
    
    @staticmethod
    def is_english_char(char):
        """Check if character is English."""
        return ord(char) < 128
