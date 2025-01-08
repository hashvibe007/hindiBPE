class HindiTokenizer:
    def __init__(self):
        # Base Varnmala (वर्णमाला)
        self.VYANJAN = set(
            [  # व्यंजन (Consonants)
                "क",
                "ख",
                "ग",
                "घ",
                "ङ",
                "च",
                "छ",
                "ज",
                "झ",
                "ञ",
                "ट",
                "ठ",
                "ड",
                "ढ",
                "ण",
                "त",
                "थ",
                "द",
                "ध",
                "न",
                "प",
                "फ",
                "ब",
                "भ",
                "म",
                "य",
                "र",
                "ल",
                "व",
                "श",
                "ष",
                "स",
                "ह",
            ]
        )

        self.SWAR = set(
            [  # स्वर (Vowels)
                "अ",
                "आ",
                "इ",
                "ई",
                "उ",
                "ऊ",
                "ऋ",
                "ए",
                "ऐ",
                "ओ",
                "औ",
            ]
        )

        self.MATRAS = set(
            [  # मात्राएँ
                "ा",
                "ि",
                "ी",
                "ु",
                "ू",
                "ृ",
                "े",
                "ै",
                "ो",
                "ौ",
            ]
        )

        self.SPECIAL_CHARS = set(
            [
                "्",  # Halant (विराम)
                "ं",  # Anusvara (अनुस्वार)
                "ः",  # Visarga (विसर्ग)
                "ँ",  # Chandrabindu (चन्द्रबिन्दु)
                "़",  # Nukta (नुक्ता)
            ]
        )

        # Combined sets for convenience
        self.CONSONANTS = self.VYANJAN  # For backward compatibility
        self.VOWELS = self.SWAR  # For backward compatibility
        self.BASE_VOCAB = self.VYANJAN | self.SWAR | self.MATRAS | self.SPECIAL_CHARS

        self.base_vocab_stats = {
            "vyanjan": len(self.VYANJAN),
            "swar": len(self.SWAR),
            "matras": len(self.MATRAS),
            "special": len(self.SPECIAL_CHARS),
            "total": len(self.BASE_VOCAB),
        }

    def _get_token_type(self, token: str) -> str:
        """Determine the type of a token"""
        if len(token) == 1:
            if token in self.VYANJAN:
                return "consonant"
            elif token in self.SWAR:
                return "vowel"
            elif token in self.MATRAS:
                return "matra"
            elif token in self.SPECIAL_CHARS:
                return "special"
        return "compound"

    def tokenize(self, text: str) -> list:
        """Basic character-level tokenization"""
        return list(text)

    def is_hindi_char(self, char: str) -> bool:
        """Check if a character is a Hindi character"""
        return char in self.BASE_VOCAB
