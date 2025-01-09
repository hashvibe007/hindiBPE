class FeedbackLoop:
    def __init__(self, target_compression_ratio):
        self.target_compression_ratio = target_compression_ratio

    def evaluate_performance(self, current_compression_ratio):
        return current_compression_ratio < self.target_compression_ratio
