from bs4 import BeautifulSoup
import os
import re
import nltk
from nltk.data import find
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import streamlit as st
import joblib
import json

# ---------------------------
# Ensure NLTK data
# ---------------------------
def ensure_nltk_data():
    resources = {
        'punkt': 'tokenizers/punkt',
        'punkt_tab': 'tokenizers/punkt_tab',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet'
    }
    for pkg, path in resources.items():
        try:
            find(path)
        except LookupError:
            nltk.download(pkg, quiet=True)

ensure_nltk_data()

stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

# ---------------------------
# Preprocessing
# ---------------------------
def clean_text(text):
    text = re.sub(r'\[\d+\]', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

def extract_text_from_html(filepath):
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')
    content_div = (
        soup.find('div', id='mw-content-text')
        or soup.find('div', id='bodyContent')
        or soup.find('body')
    )
    if not content_div:
        return ""
    paragraphs = content_div.find_all('p')
    text = " ".join(p.get_text() for p in paragraphs)
    return clean_text(text)

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    return [lemmatizer.lemmatize(t) for t in tokens]

def vectorize_text(text, vectorizer):
    tokens = preprocess(text)
    return vectorizer.transform([" ".join(tokens)])

# ---------------------------
# Load corpus & build index
# ---------------------------
METADATA_FILE = "index_metadata.json"

def cached_file_count():
    """Return the number of files used to build the cached index."""
    if not os.path.exists(METADATA_FILE):
        return 0
    with open(METADATA_FILE, "r") as f:
        data = json.load(f)
    return data.get("file_count", 0)

def cache_file_count(count):
    """Save the number of files used to build the cached index."""
    data = {"file_count": count}
    with open(METADATA_FILE, "w") as f:
        json.dump(data, f)


@st.cache_resource
def load_index(data_dir="."):
    placeholder = st.empty()

    corpus_files = [f for f in os.listdir(data_dir) if f.lower().endswith((".html", ".htm"))]
    rebuild = not os.path.exists("tfidf_index.pkl") or len(corpus_files) != cached_file_count()

    if not rebuild:
        placeholder.text("Loading cached index...")
        vectorizer, tfidf_matrix, doc_ids = joblib.load("tfidf_index.pkl")
        placeholder.text("Index loaded ✅")
        return vectorizer, tfidf_matrix, doc_ids

    corpus = {os.path.splitext(f)[0]: extract_text_from_html(os.path.join(data_dir, f))
              for f in corpus_files}
    placeholder.text(f"Building TF-IDF index for {len(corpus)} documents...")

    corpus_tokens = {doc_id: preprocess(text) for doc_id, text in corpus.items()}
    docs_as_text = [" ".join(tokens) for tokens in corpus_tokens.values()]
    doc_ids = list(corpus_tokens.keys())

    vectorizer = TfidfVectorizer(max_features=10000, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(docs_as_text)

    joblib.dump((vectorizer, tfidf_matrix, doc_ids), "tfidf_index.pkl")
    cache_file_count(len(corpus_files))

    placeholder.text(f"Index built ✅ ({len(doc_ids)} docs)")
    return vectorizer, tfidf_matrix, doc_ids

# ---------------------------
# Query expansion
# ---------------------------
def expand_query(query, mode):
    tokens = query.lower().split()
    expanded = set(tokens)

    if mode > 0:
        for word in tokens:
            # only nouns
            synsets = wn.synsets(word, pos=wn.NOUN)
            for syn in synsets:
                # filter for musical-related senses
                if "music" in syn.definition() or "musical" in syn.definition():
                    for lemma in syn.lemmas():
                        expanded.add(lemmatizer.lemmatize(lemma.name().replace("_", " ")))

                    # add hypernyms if they seem relevant
                    for hyper in syn.hypernyms():
                        if "music" in hyper.definition() or "musical" in hyper.definition():
                            for lemma in hyper.lemmas():
                                expanded.add(lemmatizer.lemmatize(lemma.name().replace("_", " ")))

    elif mode < 0:
        # narrow: keep longer words
        expanded = {w for w in tokens if len(w) > 3}

    return " ".join(expanded)

# ---------------------------
# Search function
# ---------------------------
def search(query, vectorizer, tfidf_matrix, doc_ids, top_k=5):
    query_vec = vectorize_text(query, vectorizer)
    sims = linear_kernel(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(doc_ids[i], sims[i]) for i in top_indices]

# ---------------------------
# Initialize TF-IDF index variables
# ---------------------------
vectorizer = None
tfidf_matrix = None
doc_ids = None

vectorizer, tfidf_matrix, doc_ids = load_index()

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("🔍 Wikipedia Search Demo")
query = st.text_input("Enter your query:")
mode = st.slider("Query Expansion", -1, 1, 0, help="-1 narrow, 0 normal, 1 broad")

if st.button("Search") and query.strip():
    expanded_query = expand_query(query, mode)
    st.markdown(f"**Expanded query:** {expanded_query}")

    results = search(expanded_query, vectorizer, tfidf_matrix, doc_ids)
    st.subheader("Top Results:")
    for doc, score in results:
        st.markdown(f"**{doc}** — {score:.3f}")
