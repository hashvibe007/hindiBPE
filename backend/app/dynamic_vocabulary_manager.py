class DynamicVocabularyManager:
    def __init__(self, initial_vocabulary):
        self.vocabulary = set(initial_vocabulary)

    def update_vocabulary(self, token_frequencies, threshold):
        for token, freq in token_frequencies.items():
            if freq > threshold:
                self.vocabulary.add(token)

    def get_vocabulary(self):
        return self.vocabulary
