from collections import defaultdict, Counter
from typing import Dict, List, Tuple, Set
import json
from .hindi_tokenizer import HindiTokenizer  # Import the base class
from tqdm import tqdm
import os
import time
import re
from app.adaptive_bpe import AdaptiveBPE
import logging

logging.basicConfig(
    level=logging.ERROR, format="%(asctime)s - %(levelname)s - %(message)s"
)


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

    async def learn_bpe(self, text: str, manager=None, resume_from: str = None):
        """Learn BPE merge operations with real-time updates"""
        if resume_from and os.path.exists(resume_from):
            print(f"Resuming training from {resume_from}")
            self.load_model(resume_from)
            initial_vocab_size = len(self.vocab)
        else:
            initial_vocab_size = self.initialize_vocab()

        # Track vocabulary growth details
        self.vocab_growth = {
            "tokens": [],
            "frequencies": [],
            "compositions": [],
            "merge_steps": [],
            "timestamps": [],
        }

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

        print(f"\nInitial State:")
        print(f"Base vocabulary size: {len(self.BASE_VOCAB)}")
        print(f"Starting with {len(word_freqs)} unique words")
        print(f"Total characters: {original_char_count}")
        print("\nVocabulary composition:")
        # for key, value in self.base_vocab_stats.items():
        # print(f"  {key}: {value}")

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

                # Print new token details
                if num_merges % 100 == 0:  # Print every 100 merges
                    print(f"\nMerge {num_merges + 1}:")
                    print(f"New token: '{new_token}' (freq: {frequency})")
                    print(f"Pair: '{best_pair[0][0]}' + '{best_pair[0][1]}'")
                    print(f"Token composition: {' + '.join(list(new_token))}")
                    print(f"Current vocab size: {len(self.vocab) + 1}")
                    print(f"Learned tokens: {len(self.learned_vocab)}")
                    print(f"Compression ratio: {compression_ratio:.2f}")

                    # Show example usage
                    example_words = []
                    bigram = " ".join(best_pair[0])  # Fix: Define bigram here
                    for word, freq in word_freqs.items():
                        if bigram in word:
                            example_words.append(word.replace(" ", ""))
                            if len(example_words) >= 3:
                                break
                    if example_words:
                        print("Example words:", ", ".join(example_words))

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
                    "example_words": example_words if num_merges % 100 == 0 else [],
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

                # Save checkpoint more frequently
                if num_merges % 100 == 0:
                    self._save_intermediate_vocab("bpe_model_latest.json")

                # Track vocabulary growth
                self.vocab_growth["tokens"].append(new_token)
                self.vocab_growth["frequencies"].append(frequency)
                self.vocab_growth["compositions"].append(list(best_pair[0]))
                self.vocab_growth["merge_steps"].append(num_merges)
                self.vocab_growth["timestamps"].append(time.time())

                # Adaptive BPE: Review and adjust merge operations based on token frequency
                if num_merges % 50 == 0:  # Review every 50 merges
                    print("\nReviewing vocabulary...")
                    # Example logic: prioritize tokens with higher frequency
                    frequent_tokens = [
                        token for token, freq in self.token_usage.items() if freq > 5
                    ]
                    print(f"Frequent tokens: {frequent_tokens}")

                    # Adjust merges based on frequency using AdaptiveBPE
                    adaptive_bpe = AdaptiveBPE(self.merges)
                    adaptive_bpe.perform_merges(self.token_usage)
                    self.merges = adaptive_bpe.get_merges()

                    # Update vocabulary with frequent tokens
                    for token in frequent_tokens:
                        if token not in self.vocab:
                            self.vocab.add(token)
                            print(f"Added {token} to vocabulary based on frequency.")

                    # Save updated model
                    self._save_intermediate_vocab("bpe_model_latest.json")

                # Track token frequency for Devanagari tokens only
                devanagari_pattern = re.compile(r"[\u0900-\u097F]+")
                devanagari_tokens = [
                    token
                    for token in current_tokens
                    if devanagari_pattern.fullmatch(token)
                ]
                self.token_usage.update(devanagari_tokens)

                # Save token frequencies periodically
                if num_merges % 100 == 0:
                    self.save_token_frequencies("token_frequencies.json")

                # Update vocabulary based on token frequencies and threshold
                if num_merges % 100 == 0:
                    self.update_vocabulary_based_on_frequency(threshold=5)

        print("\nFinal Training Summary:")
        print(f"Base vocabulary size: {len(self.BASE_VOCAB)}")
        print(f"Learned vocabulary size: {len(self.learned_vocab)}")
        print(f"Total vocabulary size: {len(self.vocab)}")
        print(f"Total merge operations: {len(self.merge_history)}")
        print(f"Final compression ratio: {compression_ratio:.2f}")

        return self.training_progress

    def _save_intermediate_vocab(self, filename: str):
        """Save intermediate vocabulary during training"""
        checkpoint_data = {
            "vocab": list(self.vocab),
            "learned_vocab": list(self.learned_vocab),
            "merges": {" ".join(k): v for k, v in self.merges.items()},
            "merge_history": self.merge_history,
            "base_vocab_stats": self.base_vocab_stats,
            "training_stats": {
                "total_merges": len(self.merge_history),
                "vocab_size": len(self.vocab),
                "learned_vocab_size": len(self.learned_vocab),
            },
        }

        with open(filename, "w", encoding="utf-8") as f:
            json.dump(checkpoint_data, f, ensure_ascii=False, indent=2)

        print(f"\nSaved checkpoint to {filename}")

    def assign_token_numbers(self):
        """Assign unique numbers to each token in the vocabulary"""
        # Assign numbers to base vocabulary
        self.token_numbers = {token: i + 1 for i, token in enumerate(self.BASE_VOCAB)}
        logging.info("Base vocabulary token numbers: %s", self.token_numbers)

        # Assign incremental numbers to the rest of the vocabulary
        current_number = len(self.BASE_VOCAB) + 1
        for token in sorted(self.vocab - self.BASE_VOCAB):
            self.token_numbers[token] = current_number
            current_number += 1

        # Ensure all tokens in the vocabulary are numbered
        for token in self.vocab:
            if token not in self.token_numbers:
                self.token_numbers[token] = current_number
                current_number += 1
        logging.info("Complete token numbers: %s", self.token_numbers)

    def tokenize_with_details(self, text: str) -> Dict:
        """Tokenize text and provide detailed analysis"""
        original_tokens = self.tokenize(text)  # Character-level tokenization
        bpe_tokens = self.tokenize_bpe(text)  # BPE tokenization

        # Update token usage statistics
        self.token_usage.update(bpe_tokens)

        # Calculate original character count as byte length
        original_char_count = len(list(map(int, text.encode("utf-8"))))

        # Encode BPE tokens individually
        bpe_encoded_tokens = [
            list(map(int, token.encode("utf-8"))) for token in bpe_tokens
        ]

        # Encode original tokens individually
        original_encoded_tokens = [
            list(map(int, token.encode("utf-8"))) for token in original_tokens
        ]
        bpe_char_count = len(original_encoded_tokens)

        # Assign token numbers
        self.assign_token_numbers()

        # Calculate compression ratio correctly
        compression_ratio = (
            round(original_char_count / len(bpe_tokens), 2)
            if len(bpe_tokens) > 0
            else 0
        )

        return {
            "original_text": text,
            "original_tokens": original_tokens,
            "original_encoded_tokens": original_encoded_tokens,  # Ensure this is included
            "bpe_tokens": bpe_tokens,
            "bpe_encoded_tokens": bpe_encoded_tokens,
            "token_numbers": [
                self.token_numbers.get(token, -1) for token in bpe_tokens
            ],
            "stats": {
                "original_chars": original_char_count,
                "original_token_count": original_char_count,
                "bpe_token_count": bpe_char_count,
                "compression_ratio": compression_ratio,
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
        # Define a regex pattern to match only Devanagari characters
        devanagari_pattern = re.compile(r"[\u0900-\u097F]+")

        # Split text into words and filter out non-Devanagari words
        words = text.split()
        filtered_words = [word for word in words if devanagari_pattern.fullmatch(word)]

        # Split each word into space-separated characters for BPE
        word_freqs = Counter()
        for word in filtered_words:
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
            # print(f"Processing word: {word}")  # Debug statement
            # Check if the word is already in the vocabulary
            if word in self.vocab:
                # print(f"Word '{word}' is in the vocabulary.")  # Debug statement
                result.append(word)
                continue

            # Start with character-level tokens
            word_tokens = " ".join(list(word))  # Space-separated characters
            # print(f"Initial tokens: {word_tokens}")  # Debug statement

            # Apply merges iteratively
            while True:
                # Find possible pairs
                pairs = [
                    (pair[0], pair[1])
                    for pair in zip(word_tokens.split()[:-1], word_tokens.split()[1:])
                ]
                # print(f"Possible pairs: {pairs}")  # Debug statement

                # Check if any split tokens are in the vocabulary
                split_tokens = word_tokens.split()
                if all(token in self.vocab for token in split_tokens):
                    # print(
                    #     f"All split tokens are in the vocabulary: {split_tokens}"
                    # )  # Debug statement
                    result.extend(split_tokens)
                    break

                # Find first applicable merge
                for pair in pairs:
                    if pair in self.merges:
                        new_token = self.merges[pair]
                        bigram = " ".join(pair)
                        word_tokens = word_tokens.replace(bigram, new_token)
                        # print(
                        #     f"Applied merge: {pair} -> {new_token}"
                        # )  # Debug statement
                        break
                else:
                    # No more merges possible
                    break

            # Add final tokens
            result.extend(word_tokens.split())
            # print(
            #     f"Final tokens for word '{word}': {word_tokens.split()}"
            # )  # Debug statement

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

    def save_token_frequencies(self, filename: str):
        """Save token frequencies to a JSON file"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.token_usage, f, ensure_ascii=False, indent=2)
        print(f"Token frequencies saved to {filename}")

    def update_vocabulary_based_on_frequency(self, threshold: int):
        """Update vocabulary based on token frequencies and a defined threshold"""
        # Load token frequencies from JSON file
        with open("token_frequencies.json", "r", encoding="utf-8") as f:
            token_frequencies = json.load(f)

        # Sort tokens by frequency
        sorted_tokens = sorted(
            token_frequencies.items(), key=lambda x: x[1], reverse=True
        )

        # Update vocabulary based on threshold
        for token, freq in sorted_tokens:
            if freq >= threshold:
                if token not in self.vocab:
                    self.vocab.add(token)
                    print(f"Added {token} to vocabulary based on frequency {freq}.")

        # Save updated model
        self._save_intermediate_vocab("bpe_model_latest.json")

    def update_bpe_model_from_frequencies(self, threshold: int):
        """One-time update of BPE model using token frequencies"""
        # Load token frequencies from JSON file
        with open("token_frequencies.json", "r", encoding="utf-8") as f:
            token_frequencies = json.load(f)

        # Sort tokens by frequency in descending order
        sorted_tokens = sorted(
            token_frequencies.items(), key=lambda x: x[1], reverse=True
        )

        # Clear existing vocabulary
        updated_vocab = set()

        # Update vocabulary based on top 5000 most frequent tokens, skipping base vocabulary
        top_tokens = [
            token for token, freq in sorted_tokens if token not in self.BASE_VOCAB
        ][:5000]
        for token in top_tokens:
            updated_vocab.add(token)
            print(f"Added {token} to vocabulary based on frequency.")

        # Save updated model
        model_data = {
            "vocab": list(updated_vocab),
            "merges": {},  # Clear merges as well
            "merge_history": [],
            "base_vocab_stats": {},
            "training_stats": {
                "total_merges": 0,
                "vocab_size": len(updated_vocab),
                "learned_vocab_size": 0,
            },
        }
        with open("bpe_model_latest.json", "w", encoding="utf-8") as f:
            json.dump(model_data, f, ensure_ascii=False, indent=2)
        print("Updated BPE model saved to bpe_model_latest.json")

        # Save sorted token frequencies back to JSON file
        sorted_token_frequencies = {token: freq for token, freq in sorted_tokens}
        with open("token_frequencies.json", "w", encoding="utf-8") as f:
            json.dump(sorted_token_frequencies, f, ensure_ascii=False, indent=2)
        print("Sorted token frequencies saved to token_frequencies.json")
