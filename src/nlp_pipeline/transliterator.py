class Transliterator:
    """Transliterate between Sinhala, Tamil, and Roman scripts."""
    
    @staticmethod
    def sinhala_to_roman(text):
        """Convert Sinhala script to Roman."""
        sinhala_to_roman_map = {
            'ක': 'ka', 'ඛ': 'kha', 'ග': 'ga', 'ඝ': 'gha',
            'ච': 'ca', 'ඡ': 'cha', 'ජ': 'ja', 'ඣ': 'jha',
        }
        result = []
        for char in text:
            if char in sinhala_to_roman_map:
                result.append(sinhala_to_roman_map[char])
            else:
                result.append(char)
        return ''.join(result)
    
    @staticmethod
    def tamil_to_roman(text):
        """Convert Tamil script to Roman."""
        tamil_to_roman_map = {
            'க': 'ka', 'ங': 'nga', 'ச': 'cha', 'ஜ': 'ja',
            'ட': 'ta', 'ண': 'na', 'த': 'tha', 'ந': 'na',
        }
        result = []
        for char in text:
            if char in tamil_to_roman_map:
                result.append(tamil_to_roman_map[char])
            else:
                result.append(char)
        return ''.join(result)