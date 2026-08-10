class Transliterator:
    """Transliterate between Sinhala, Tamil, and Roman scripts."""
    
    @staticmethod
    def sinhala_to_roman(text):
        """Convert Sinhala script to Roman."""
        sinhala_to_roman_map = {
            'ක': 'ka', 'ඛ': 'kha', 'ග': 'ga', 'ඝ': 'gha',
            'ච': 'ca', 'ඡ': 'cha', 'ජ': 'ja', 'ඣ': 'jha',
            'ට': 'ta', 'ඨ': 'tha', 'ඩ': 'da', 'ඪ': 'dha',
            'ත': 'tha', 'ථ': 'tha', 'ද': 'da', 'ධ': 'dha',
            'න': 'na', 'ප': 'pa', 'ඵ': 'pha', 'බ': 'ba',
            'භ': 'bha', 'ම': 'ma', 'ය': 'ya', 'ර': 'ra',
            'ල': 'la', 'ව': 'va', 'ශ': 'sha', 'ෂ': 'sha',
            'ස': 'sa', 'හ': 'ha', 'ළ': 'la', 'ඤ': 'na',
            'ඥ': 'na', 'ඳ': 'nda', 'ඹ': 'mba',
            'අ': 'a', 'ආ': 'aa', 'ඇ': 'ae', 'ඈ': 'aee',
            'ඉ': 'i', 'ඊ': 'ee', 'උ': 'u', 'ඌ': 'oo',
            'ඍ': 'ru', 'ඎ': 'ruu', 'ඏ': 'lu', 'ඐ': 'luu',
            'එ': 'e', 'ඒ': 'ei', 'ඓ': 'ai', 'ඔ': 'o',
            'ඕ': 'ou', 'ඖ': 'au'
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
        """Convert Tamil script to Roman (ISO 15919 standard)."""
        tamil_to_roman_map = {
            'அ': 'a', 'ஆ': 'ā', 'இ': 'i', 'ஈ': 'ī',
            'உ': 'u', 'ஊ': 'ū', 'எ': 'e', 'ஏ': 'ē',
            'ஐ': 'ai', 'ஒ': 'o', 'ஓ': 'ō', 'ஔ': 'au',
            'க': 'k', 'ங': 'ṅ', 'ச': 'c', 'ஜ': 'j',
            'ஞ': 'ñ', 'ட': 'ṭ', 'ண': 'ṇ', 'த': 't',
            'ந': 'n', 'ப': 'p', 'ம': 'm', 'ய': 'y',
            'ர': 'r', 'ல': 'l', 'வ': 'v', 'ழ': 'ḻ',
            'ள': 'ḷ', 'ற': 'ṟ', 'ன': 'ṉ', 'ஹ': 'h',
            'க்': 'k', 'ச்': 'c', 'ட்': 'ṭ', 'த்': 't',
            'ப்': 'p', 'ற்': 'ṟ', 'ன்': 'ṉ', 'ன்': 'n',
            'ம்': 'm', 'ய்': 'y', 'ர்': 'r', 'ல்': 'l',
            'ழ்': 'ḻ', 'ள்': 'ḷ', 'ஸ': 's', 'ஷ': 'ṣ'
        }
        result = []
        for char in text:
            if char in tamil_to_roman_map:
                result.append(tamil_to_roman_map[char])
            else:
                result.append(char)
        return ''.join(result)
    
    @staticmethod
    def detect_and_transliterate(text):
        """Detect language and transliterate accordingly."""
        # Simple detection based on Unicode ranges
        sinhala_count = 0
        tamil_count = 0
        
        for char in text:
            if 0x0D80 <= ord(char) <= 0x0DFF:
                sinhala_count += 1
            elif 0x0B80 <= ord(char) <= 0x0BFF:
                tamil_count += 1
        
        if sinhala_count > tamil_count:
            return Transliterator.sinhala_to_roman(text)
        elif tamil_count > sinhala_count:
            return Transliterator.tamil_to_roman(text)
        else:
            return text
