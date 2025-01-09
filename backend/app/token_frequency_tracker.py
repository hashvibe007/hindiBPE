from collections import defaultdict


class TokenFrequencyTracker:
    def __init__(self):
        self.token_frequencies = defaultdict(int)

    def update_frequencies(self, tokens):
        for token in tokens:
            self.token_frequencies[token] += 1

    def get_frequencies(self):
        return dict(self.token_frequencies)
