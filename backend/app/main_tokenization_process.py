import re
from app.token_frequency_tracker import TokenFrequencyTracker
from app.dynamic_vocabulary_manager import DynamicVocabularyManager
from app.adaptive_bpe import AdaptiveBPE
from app.feedback_loop import FeedbackLoop


def preprocess_text(text):
    # Define a regex pattern to match only Devanagari characters and numbers
    devanagari_pattern = re.compile(r"[\u0900-\u097F]+")
    number_pattern = re.compile(r"\d{3,}")

    # Find all Devanagari words and sequences of three or more numbers
    words = devanagari_pattern.findall(text)
    numbers = number_pattern.findall(text)

    # Combine Devanagari words and number sequences
    return words + numbers


def calculate_compression_ratio(original_text, tokens):
    return len(original_text) / len(" ".join(tokens))


def main_tokenization_process(
    texts,
    initial_vocabulary,
    initial_merges,
    target_compression_ratio,
    frequency_threshold,
):
    frequency_tracker = TokenFrequencyTracker()
    vocabulary_manager = DynamicVocabularyManager(initial_vocabulary)
    bpe = AdaptiveBPE(initial_merges)
    feedback_loop = FeedbackLoop(target_compression_ratio)

    for text in texts:
        # Preprocess text to extract relevant tokens
        tokens = preprocess_text(text)
        frequency_tracker.update_frequencies(tokens)
        token_frequencies = frequency_tracker.get_frequencies()
        vocabulary_manager.update_vocabulary(token_frequencies, frequency_threshold)
        bpe.perform_merges(token_frequencies)
        current_compression_ratio = calculate_compression_ratio(text, tokens)
        if feedback_loop.evaluate_performance(current_compression_ratio):
            pass

    return vocabulary_manager.get_vocabulary(), bpe.get_merges()
