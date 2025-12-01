from bs4 import BeautifulSoup
import os
import re
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import streamlit as st
import joblib
import json
import urllib.parse

# ---------------------------
# Setup
# ---------------------------
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)

# Download stopwords and wordnet if missing
import nltk
nltk.data.path.append(NLTK_DATA_DIR)
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------------------
# Preprocessing
# ---------------------------
def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)  # remove [1], [2]
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_from_html(filepath):
    with open(filepath, "r", encoding="utf-8") as f:
        soup = BeautifulSoup(f, "html.parser")
    content_div = (
        soup.find("div", id="mw-content-text")
        or soup.find("div", id="bodyContent")
        or soup.find("body")
    )
    if not content_div:
        return ""
    paragraphs = content_div.find_all(["p", "h1", "h2", "h3"])
    text = " ".join(p.get_text() for p in paragraphs)
    return clean_text(text)

def preprocess(text):
    # simple regex-based tokenizer instead of punkt
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    tokens = [t for t in tokens if t not in stop_words]
    return [lemmatizer.lemmatize(t) for t in tokens]

def vectorize_text(text, vectorizer):
    tokens = preprocess(text)
    return vectorizer.transform([" ".join(tokens)])

# ---------------------------
# Index caching
# ---------------------------
METADATA_FILE = "index_metadata.json"

def cached_file_count():
    if not os.path.exists(METADATA_FILE):
        return 0
    with open(METADATA_FILE, "r") as f:
        data = json.load(f)
    return data.get("file_count", 0)

def cache_file_count(count):
    with open(METADATA_FILE, "w") as f:
        json.dump({"file_count": count}, f)

@st.cache_resource
def load_index(data_dir="articles"):
    placeholder = st.empty()
    corpus_files = [f for f in os.listdir(data_dir) if f.lower().endswith((".html", ".htm"))]
    rebuild = not os.path.exists("tfidf_index.pkl") or len(corpus_files) != cached_file_count()

    if not rebuild:
        placeholder.text("Loading cached index...")
        vectorizer, tfidf_matrix, doc_ids, doc_paths = joblib.load("tfidf_index.pkl")
        placeholder.text("Index loaded ✅")
        return vectorizer, tfidf_matrix, doc_ids, doc_paths

    placeholder.text(f"Building TF-IDF index for {len(corpus_files)} documents...")
    corpus = {
        os.path.splitext(f)[0]: extract_text_from_html(os.path.join(data_dir, f))
        for f in corpus_files
    }

    corpus_tokens = {doc_id: preprocess(text) for doc_id, text in corpus.items()}
    docs_as_text = [" ".join(tokens) for tokens in corpus_tokens.values()]
    doc_ids = list(corpus_tokens.keys())

    vectorizer = TfidfVectorizer(max_features=10000, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(docs_as_text)

    doc_paths = {os.path.splitext(f)[0]: os.path.join(data_dir, f) for f in corpus_files}

    joblib.dump((vectorizer, tfidf_matrix, doc_ids, doc_paths), "tfidf_index.pkl")
    cache_file_count(len(corpus_files))
    placeholder.text(f"Index built ✅ ({len(doc_ids)} docs)")
    return vectorizer, tfidf_matrix, doc_ids, doc_paths

# ---------------------------
# Query expansion
# ---------------------------
def expand_query(query, mode, vectorizer_vocab=None, max_expansions=5):
    tokens = preprocess(query)
    if not tokens:
        return query

    if vectorizer_vocab is None:
        vectorizer_vocab = set()

    expanded = set(tokens)

    if mode == 0:
        return " ".join(tokens)
    if mode < 0:
        narrowed = {w for w in tokens if len(w) > 3}
        return " ".join(narrowed)

    for word in tokens:
        word_expansions = set()
        for pos in [wn.NOUN, wn.VERB, wn.ADJ]:
            synsets = wn.synsets(word, pos=pos)
            for syn in synsets:
                for lemma in syn.lemmas():
                    lemma_name = lemma.name().replace("_", " ").lower()
                    word_expansions.add(lemmatizer.lemmatize(lemma_name))
                for hyper in syn.hypernyms():
                    for lemma in hyper.lemmas():
                        lemma_name = lemma.name().replace("_", " ").lower()
                        word_expansions.add(lemmatizer.lemmatize(lemma_name))
        if vectorizer_vocab:
            word_expansions = {w for w in word_expansions if w in vectorizer_vocab}
        expanded.update(list(word_expansions)[:max_expansions])

    weighted_query = tokens + tokens + list(expanded)
    return " ".join(weighted_query)

# ---------------------------
# Search
# ---------------------------
def search(query, vectorizer, tfidf_matrix, doc_ids, top_k=5):
    query_vec = vectorize_text(query, vectorizer)
    sims = linear_kernel(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(doc_ids[i], sims[i]) for i in top_indices]

# ---------------------------
# Initialize index
# ---------------------------
vectorizer, tfidf_matrix, doc_ids, doc_paths = load_index()

# ---------------------------
# Streamlit UI with Enter-to-Submit
# ---------------------------
st.title("🔍 Wikipedia Search Demo")

with st.form("search_form"):
    query = st.text_input("Enter your query:")
    mode = st.slider("Query Expansion", -1, 1, 0, help="-1 narrow, 0 normal, 1 broad")
    submitted = st.form_submit_button("Search")

if submitted and query.strip():
    expanded_query = expand_query(query, mode)
    st.markdown(f"**Expanded query:** {expanded_query}")

    results = search(expanded_query, vectorizer, tfidf_matrix, doc_ids)
    st.subheader("Top Results:")
    github_pages = "https://cwilburn-dev.github.io/INFO556Project/articles/"

    for doc, score in results:
        encoded_filename = urllib.parse.quote(f"{doc}.htm")
        file_url = f"{github_pages}/{encoded_filename}"
        st.markdown(f"[{doc}]({file_url}) — {score:.3f}")

