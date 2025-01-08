import requests
import wikitextparser as wtp
from bs4 import BeautifulSoup
import re
import os
from tqdm import tqdm


def download_hindi_wikipedia_content(num_articles=500, sentences_per_article=10):
    """Download Hindi articles from Wikipedia"""

    # Hindi Wikipedia API endpoint
    api_url = "https://hi.wikipedia.org/w/api.php"

    print("Downloading Hindi Wikipedia articles...")

    # Parameters for the API request
    params = {
        "action": "query",
        "format": "json",
        "list": "random",
        "rnlimit": num_articles,
        "rnnamespace": 0,  # Main namespace
    }

    # Get random article titles
    response = requests.get(api_url, params=params)
    articles = response.json()["query"]["random"]

    collected_sentences = []

    for article in tqdm(articles, desc="Processing articles"):
        # Get article content
        params = {
            "action": "parse",
            "format": "json",
            "page": article["title"],
            "prop": "text",
        }

        try:
            response = requests.get(api_url, params=params)
            content = response.json()["parse"]["text"]["*"]

            # Parse HTML content
            soup = BeautifulSoup(content, "html.parser")
            text = soup.get_text()

            # Clean text
            text = clean_text(text)

            # Split into sentences (basic splitting by ।, ?, !)
            sentences = re.split("[।?!]", text)
            sentences = [
                s.strip() for s in sentences if len(s.strip()) > 20
            ]  # Min length filter

            # Add top sentences from this article
            collected_sentences.extend(sentences[:sentences_per_article])

        except Exception as e:
            print(f"Error processing article {article['title']}: {str(e)}")
            continue

    return collected_sentences


def clean_text(text: str) -> str:
    """Clean the Wikipedia text"""
    # Remove references [1], [2], etc.
    text = re.sub(r"\[\d+\]", "", text)

    # Remove multiple spaces
    text = re.sub(r"\s+", " ", text)

    # Remove special Wikipedia markup
    text = re.sub(r"\{\{.*?\}\}", "", text)
    text = re.sub(r"\[\[(?:[^|\]]*\|)?([^\]]+)\]\]", r"\1", text)

    # Remove empty brackets
    text = re.sub(r"\(\s*\)", "", text)

    # Remove URLs
    text = re.sub(r"http\S+", "", text)

    return text.strip()


def save_sentences(sentences: list, output_file: str):
    """Save sentences to a file"""
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    with open(output_file, "w", encoding="utf-8") as f:
        for sentence in sentences:
            f.write(sentence + "\n")


def main():
    # Create data directory if it doesn't exist
    os.makedirs("data", exist_ok=True)

    print("Downloading Hindi text data...")
    sentences = download_hindi_wikipedia_content(
        num_articles=500, sentences_per_article=20
    )

    # Save raw sentences
    output_file = "data/hindi_wiki_corpus.txt"
    save_sentences(sentences, output_file)

    print(f"\nData collection complete!")
    print(f"Total sentences collected: {len(sentences)}")
    print(f"Data saved to: {output_file}")

    # Show sample statistics
    total_chars = sum(len(s) for s in sentences)
    print(f"\nDataset Statistics:")
    print(f"Total sentences: {len(sentences)}")
    print(f"Total characters: {total_chars}")
    print(f"Average sentence length: {total_chars/len(sentences):.2f} characters")

    # Show sample
    print("\nSample sentences:")
    for i, sentence in enumerate(sentences[:5]):
        print(f"{i+1}. {sentence}")


if __name__ == "__main__":
    main()
