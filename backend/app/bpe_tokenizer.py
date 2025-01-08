from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import json
from .hindi_tokenizer import HindiTokenizer  # Import the base class
from tqdm import tqdm


class BPETokenizer(HindiTokenizer):
    def __init__(self, vocab_size=5000):
        # Call parent class's __init__ first to initialize BASE_VOCAB
        HindiTokenizer.__init__(self)  # or super().__init__()

        # Initialize BPE-specific attributes
        self.vocab_size = vocab_size
        self.merges = {}  # Store merge operations
        self.vocab = set()  # Final vocabulary
        self.merge_history = []  # Track merge operations
        self.token_usage = Counter()  # Track token usage
        self.pair_frequencies = defaultdict(int)  # Track pair frequencies
        self.learned_vocab = set()  # Track learned tokens separately

    def initialize_vocab(self):
        """Initialize vocabulary with basic Hindi characters"""
        self.vocab = self.BASE_VOCAB.copy()  # Now BASE_VOCAB will be available
        self.learned_vocab = set()  # Reset learned tokens
        return len(self.vocab)

    def _get_pair_frequencies(self, word_freqs: Dict[str, int]) -> Dict[tuple, int]:
        """Count frequencies of adjacent pairs"""
        pairs = defaultdict(int)
        for word, freq in word_freqs.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pair = (symbols[i], symbols[i + 1])
                pairs[pair] += freq
                self.pair_frequencies[pair] += freq
        return pairs

    def learn_bpe(self, text: str):
        """Learn BPE merge operations with detailed tracking"""
        initial_vocab_size = self.initialize_vocab()

        # Track learning progress
        self.training_progress = {
            "base_vocab_stats": self.base_vocab_stats,
            "initial_vocab_size": initial_vocab_size,
            "target_vocab_size": self.vocab_size,
            "steps": [],
            "metrics": {
                "vocab_sizes": [initial_vocab_size],
                "learned_vocab_sizes": [0],
                "compression_ratios": [],
                "merge_frequencies": [],
                "unique_tokens": [initial_vocab_size],
            },
        }

        print("Preparing word frequencies...")
        word_freqs = self._get_word_frequencies(text)
        num_merges = 0
        original_char_count = sum(
            len("".join(word.split())) * freq for word, freq in word_freqs.items()
        )

        print(f"Starting with {len(word_freqs)} unique words")
        print(f"Total characters: {original_char_count}")

        with tqdm(
            total=self.vocab_size - initial_vocab_size, desc="Learning merges"
        ) as pbar:
            while len(self.vocab) < self.vocab_size:
                pairs = self._get_pair_frequencies(word_freqs)
                if not pairs:
                    print("\nNo more pairs to merge!")
                    break

                # Get most frequent pair
                best_pair = max(pairs.items(), key=lambda x: x[1])
                new_token = "".join(best_pair[0])
                frequency = best_pair[1]

                if frequency < 2:  # Stop if pairs aren't frequent enough
                    print("\nNo more frequent pairs to merge!")
                    break

                # Track metrics
                current_tokens = self.tokenize_bpe(text)
                compression_ratio = original_char_count / len(current_tokens)

                # Add to learned vocabulary
                self.learned_vocab.add(new_token)

                # Update progress metrics
                self.training_progress["metrics"]["vocab_sizes"].append(
                    len(self.vocab) + 1
                )
                self.training_progress["metrics"]["learned_vocab_sizes"].append(
                    len(self.learned_vocab)
                )
                self.training_progress["metrics"]["compression_ratios"].append(
                    compression_ratio
                )
                self.training_progress["metrics"]["merge_frequencies"].append(frequency)
                self.training_progress["metrics"]["unique_tokens"].append(
                    len(set(current_tokens))
                )

                # Track merge operation
                merge_info = {
                    "step": num_merges + 1,
                    "pair": best_pair[0],
                    "new_token": new_token,
                    "frequency": frequency,
                    "vocab_size": len(self.vocab) + 1,
                    "learned_vocab_size": len(self.learned_vocab),
                    "compression_ratio": compression_ratio,
                }
                self.merge_history.append(merge_info)
                self.training_progress["steps"].append(merge_info)

                # Add to vocabulary and merges
                self.vocab.add(new_token)
                self.merges[best_pair[0]] = new_token

                # Update word frequencies with merged pair
                word_freqs = self._apply_merge(word_freqs, best_pair[0])
                num_merges += 1
                pbar.update(1)

                # Print progress every 100 merges
                if num_merges % 100 == 0:
                    print(f"\nMerge {num_merges}:")
                    print(f"New token: {new_token} (frequency: {frequency})")
                    print(f"Vocabulary size: {len(self.vocab)}")
                    print(f"Learned tokens: {len(self.learned_vocab)}")
                    print(f"Compression ratio: {compression_ratio:.2f}")

        return self.training_progress

    def tokenize_with_details(self, text: str) -> Dict:
        """Tokenize text and provide detailed analysis"""
        original_tokens = self.tokenize(text)  # Character-level tokenization
        bpe_tokens = self.tokenize_bpe(text)  # BPE tokenization

        # Update token usage statistics
        self.token_usage.update(bpe_tokens)

        return {
            "original_text": text,
            "original_tokens": original_tokens,
            "bpe_tokens": bpe_tokens,
            "stats": {
                "original_chars": len(text.replace(" ", "")),
                "original_token_count": len(original_tokens),
                "bpe_token_count": len(bpe_tokens),
                "compression_ratio": round(
                    len(text.replace(" ", "")) / len(bpe_tokens), 2
                ),
                "unique_tokens": len(set(bpe_tokens)),
            },
            "token_details": [
                {
                    "token": token,
                    "frequency": self.token_usage[token],
                    "length": len(token),
                    "type": self._get_token_type(token),
                }
                for token in bpe_tokens
            ],
        }

    def _get_word_frequencies(self, text: str) -> Dict[str, int]:
        """Get word frequencies from text with proper character-level splitting"""
        words = text.split()
        # Split each word into space-separated characters for BPE
        word_freqs = Counter()
        for word in words:
            # Convert word to space-separated characters
            char_seq = " ".join(list(word))
            word_freqs[char_seq] += 1
        return word_freqs

    def _apply_merge(
        self, word_freqs: Dict[str, int], pair: Tuple[str, str]
    ) -> Dict[str, int]:
        """Apply a merge operation to all words"""
        new_word_freqs = {}
        bigram = " ".join(pair)  # Space between characters
        replacement = "".join(pair)  # No space in replacement

        for word, freq in word_freqs.items():
            if bigram in word:
                new_word = word.replace(bigram, replacement)
                new_word_freqs[new_word] = freq
            else:
                new_word_freqs[word] = freq

        return new_word_freqs

    def tokenize_bpe(self, text: str) -> List[str]:
        """Tokenize text using learned BPE merges"""
        words = text.split()
        result = []

        for word in words:
            # Start with character-level tokens
            word_tokens = " ".join(list(word))  # Space-separated characters

            # Apply merges iteratively
            while True:
                # Find possible pairs
                pairs = [
                    (pair[0], pair[1])
                    for pair in zip(word_tokens.split()[:-1], word_tokens.split()[1:])
                ]

                # Find first applicable merge
                for pair in pairs:
                    if pair in self.merges:
                        new_token = self.merges[pair]
                        bigram = " ".join(pair)
                        word_tokens = word_tokens.replace(bigram, new_token)
                        break
                else:
                    # No more merges possible
                    break

            # Add final tokens
            result.extend(word_tokens.split())

        return result

    def load_model(self, model_file: str):
        """Load trained BPE model from file"""
        try:
            with open(model_file, "r", encoding="utf-8") as f:
                model_data = json.load(f)

            self.vocab = set(model_data["vocab"])
            self.merges = {tuple(k.split()): v for k, v in model_data["merges"].items()}
            self.merge_history = model_data["merge_history"]

            # Initialize learned vocabulary
            self.learned_vocab = set(self.vocab) - self.BASE_VOCAB

            # Initialize training progress
            self.training_progress = {
                "base_vocab_stats": self.base_vocab_stats,
                "initial_vocab_size": len(self.BASE_VOCAB),
                "target_vocab_size": self.vocab_size,
                "steps": self.merge_history,
                "metrics": {
                    "vocab_sizes": [
                        len(self.BASE_VOCAB) + i
                        for i in range(len(self.merge_history) + 1)
                    ],
                    "learned_vocab_sizes": [
                        i for i in range(len(self.merge_history) + 1)
                    ],
                    "compression_ratios": [
                        m.get("compression_ratio", 1.0) for m in self.merge_history
                    ],
                    "merge_frequencies": [
                        m.get("frequency", 0) for m in self.merge_history
                    ],
                    "unique_tokens": [
                        len(self.BASE_VOCAB) + i
                        for i in range(len(self.merge_history) + 1)
                    ],
                },
            }

            print(f"Loaded model with vocabulary size: {len(self.vocab)}")
            print(f"Base vocabulary: {len(self.BASE_VOCAB)}")
            print(f"Learned tokens: {len(self.learned_vocab)}")
            return True
        except Exception as e:
            print(f"Error loading model: {str(e)}")
            return False
