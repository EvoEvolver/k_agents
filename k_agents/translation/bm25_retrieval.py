import re

import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from rank_bm25 import BM25Okapi
import string


# Download necessary NLTK resources
nltk.download('punkt_tab')
nltk.download('punkt')
nltk.download('stopwords')


# Initialize stopwords and punctuation
stop_words = set(stopwords.words('english'))
punctuations = set(string.punctuation)

def decompose_camel(s):
    return re.sub(r'(?<!^)(?=[A-Z])', ' ', s)

def tokenize(text):
    """Tokenizes and preprocesses text."""
    tokens = word_tokenize(text.lower())  # Lowercasing and tokenization
    tokens = [word for word in tokens if word not in stop_words and word not in punctuations]  # Remove stopwords & punctuation
    return tokens

def get_bm25_score(embed_src: list[str], stimuli_list: list[str]):
    """Calculates BM25 score between embed_src and stimuli_list."""
    # Tokenize the stimuli
    tokenized_stimuli = [tokenize(decompose_camel(stimuli)) for stimuli in stimuli_list]
    # Tokenize the embed_src
    tokenized_embed_src = [tokenize(decompose_camel(src)) for src in embed_src]
    # Initialize BM25 object
    bm25 = BM25Okapi(tokenized_embed_src)
    # Calculate BM25 scores
    scores = []
    for stimuli in tokenized_stimuli:
        score = bm25.get_scores(stimuli)
        scores.append(max(score))
    return scores


if __name__ == '__main__1':
    documents = [
        "The quick brown fox jumps over the lazy dog.",
        "A fast brown dog jumps over a sleepy fox!",
        "The dog is quick and brown.",
        "Foxes are generally quick and agile."
    ]

    query = "The quick brown fox jumps over the lazy dog."
    scores = get_bm25_score(documents, [query])
    print(scores)


if __name__ == '__main__':
    query = "ConditionalStarkShiftContinuous"
    tokens = tokenize(decompose_camel(query))
    print(tokens)