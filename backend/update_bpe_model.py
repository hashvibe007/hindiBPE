from app.bpe_tokenizer import BPETokenizer


def main():
    # Instantiate the BPETokenizer
    tokenizer = BPETokenizer()

    # Define the frequency threshold
    threshold = 10000

    # Perform the one-time update of the BPE model
    tokenizer.update_bpe_model_from_frequencies(threshold)


if __name__ == "__main__":
    main()
