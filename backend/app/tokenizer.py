from typing import List, Dict, Set
import regex as re


class HindiTokenizer:
    def __init__(self):
        # Devanagari Unicode ranges
        self.DEVANAGARI_RANGE = "\u0900-\u097f"

        # Basic character sets
        self.CONSONANTS = set("कखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह")
        self.VOWELS = set("अआइईउऊऋएऐओऔ")
        self.MATRAS = set("ािीुूृेैोौ")
        self.SPECIAL_CHARS = set("्ंःँ़")
        self.NUMERALS = set("०१२३४५६७८९")

        # Special tokens
        self.EOW_TOKEN = "</w>"  # End of word
        self.UNK_TOKEN = "<unk>"  # Unknown token
        self.PAD_TOKEN = "<pad>"  # Padding token

    def clean_text(self, text: str) -> str:
        """Clean and normalize Hindi text"""
        # Normalize Unicode
        text = self._normalize_unicode(text)

        # Remove excessive spaces
        text = re.sub(r"\s+", " ", text)

        # Keep only Devanagari, English letters, numbers and basic punctuation
        pattern = f"[^{self.DEVANAGARI_RANGE}a-zA-Z0-9\s\.\,\?\!]"
        text = re.sub(pattern, "", text)

        return text.strip()

    def _normalize_unicode(self, text: str) -> str:
        """Normalize Unicode characters to NFC form"""
        import unicodedata

        return unicodedata.normalize("NFC", text)

    def tokenize(self, text: str) -> List[str]:
        """Tokenize text into subwords"""
        chars = []
        i = 0
        text = self.clean_text(text)

        while i < len(text):
            if text[i].isspace():
                # Skip spaces in tokenization
                i += 1
                continue

            if i + 1 < len(text) and text[i + 1] in self.MATRAS:
                # Handle consonant + matra (like सु, की)
                chars.append(text[i : i + 2])
                i += 2
            elif i + 1 < len(text) and text[i + 1] == "ं":
                # Handle anusvara (like नीं)
                if i > 0 and text[i] in self.MATRAS:
                    # Combine with previous token if it ends with a matra
                    chars[-1] = chars[-1] + text[i + 1]
                else:
                    chars.append(text[i : i + 2])
                i += 2
            else:
                chars.append(text[i])
                i += 1

        return chars

    def get_stats(self, text: str) -> Dict:
        """Get tokenization statistics"""
        tokens = self.tokenize(text)
        # Count original characters excluding spaces
        orig_chars = len([c for c in text if not c.isspace()])

        return {
            "original_chars": orig_chars,
            "token_count": len(tokens),
            "unique_tokens": len(set(tokens)),
            "compression_ratio": round(orig_chars / len(tokens), 2),
        }

    def _get_token_type(self, token: str) -> str:
        """Determine token type"""
        if len(token) == 1:
            if token in self.CONSONANTS:
                return "consonant"
            elif token in self.VOWELS:
                return "vowel"
            elif token in self.MATRAS:
                return "matra"
            elif token in self.SPECIAL_CHARS:
                return "special"
        else:
            return "compound"
