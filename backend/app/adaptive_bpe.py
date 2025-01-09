class AdaptiveBPE:
    def __init__(self, initial_merges):
        self.merges = initial_merges

    def perform_merges(self, token_frequencies):
        # Analyze token frequencies to determine merge operations
        print("Analyzing token frequencies for adaptive merges...")

        # Example logic: prioritize pairs with highest combined frequency
        pair_frequencies = {}
        for token, freq in token_frequencies.items():
            # Split token into pairs
            for i in range(len(token) - 1):
                pair = (token[i], token[i + 1])
                if pair in pair_frequencies:
                    pair_frequencies[pair] += freq
                else:
                    pair_frequencies[pair] = freq

        # Sort pairs by frequency
        sorted_pairs = sorted(
            pair_frequencies.items(), key=lambda x: x[1], reverse=True
        )

        # Update merges with top pairs
        for pair, freq in sorted_pairs[:10]:  # Limit to top 10 pairs for example
            self.merges[pair] = "".join(pair)
            print(f"Merged pair {pair} with frequency {freq}")

    def get_merges(self):
        return self.merges
