import asyncio
from app.bpe_tokenizer import BPETokenizer
import json
from tqdm import tqdm


def load_sample_data(file_path: str, max_sentences: int = None) -> str:
    """Load sentences from file with progress bar"""
    sentences = []
    total_chars = 0

    print(f"Loading data from {file_path}...")

    # First count total lines for progress bar
    with open(file_path, "r", encoding="utf-8") as f:
        total_lines = sum(1 for _ in f)

    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, total=total_lines, desc="Loading sentences"):
            line = line.strip()
            if line:
                sentences.append(line)
                total_chars += len(line)
                if max_sentences and len(sentences) >= max_sentences:
                    break

    print(f"\nLoaded {len(sentences)} sentences")
    print(f"Total characters: {total_chars}")
    print(f"Average sentence length: {total_chars/len(sentences):.2f} characters")

    return "\n".join(sentences)


async def train_and_save_bpe(text: str, vocab_size: int = 5000, manager=None):
    tokenizer = BPETokenizer(vocab_size=vocab_size)
    print("\nInitializing vocabulary...")
    tokenizer.initialize_vocab()

    print("\nStarting BPE training...")
    await tokenizer.learn_bpe(text, manager=manager)

    # Save final model
    model_data = {
        "vocab": list(tokenizer.vocab),
        "merges": {" ".join(k): v for k, v in tokenizer.merges.items()},
        "merge_history": tokenizer.merge_history,
        "base_vocab_stats": tokenizer.base_vocab_stats,
        "training_stats": {
            "total_merges": len(tokenizer.merge_history),
            "final_vocab_size": len(tokenizer.vocab),
            "learned_vocab_size": len(tokenizer.learned_vocab),
        },
    }

    with open("bpe_model.json", "w", encoding="utf-8") as f:
        json.dump(model_data, f, ensure_ascii=False, indent=2)

    return tokenizer


async def main():
    # Load the larger dataset
    text = load_sample_data("data/hindi_wiki_corpus.txt", max_sentences=10000)

    # Train BPE with larger vocabulary
    tokenizer = await train_and_save_bpe(text, vocab_size=10000, manager=None)

    # Show sample tokenization
    print("\nSample Tokenization:")
    sample_texts = ["नमस्ते भारत", "मैं हिंदी सीख रहा हूं", "यह एक बहुत अच्छा दिन है"]

    for sample_text in sample_texts:
        result = tokenizer.tokenize_with_details(sample_text)
        print(f"\nInput text: {sample_text}")
        print(f"Original tokens: {result['original_tokens']}")
        print(f"BPE tokens: {result['bpe_tokens']}")
        print(f"Compression ratio: {result['stats']['compression_ratio']}:1")


if __name__ == "__main__":
    asyncio.run(main())
