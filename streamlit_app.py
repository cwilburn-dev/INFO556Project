from __future__ import annotations
from bs4 import BeautifulSoup
import os
import re
from typing import Dict, List, Tuple, Set, Optional
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import linear_kernel
import numpy as np
import streamlit as st
import joblib
import json
import urllib.parse
import nltk

# region CONSTANTS/CONFIG
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)

for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)

stop_words: Set[str] = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# color constants for UI
COLOR_ORIGINAL = "#4DA6FF"  # Blue
COLOR_CORE = "#FF4D4D"      # Red
COLOR_EXPANDED = "#33CC33"  # Green
COLOR_OTHER = "#FFFFFF"     # White

METADATA_FILE = "index_metadata.json"
# endregion

# region PREPROCESSING
def clean_text(text: str) -> str:
    """Remove bracketed citations and normalize whitespace."""
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def extract_text_from_html(filepath: str) -> str:
    """Extract content text from a Wikipedia HTML file."""
    if not os.path.exists(filepath):
        return ""

    try:
        with open(filepath, "r", encoding="utf-8") as f:
            soup = BeautifulSoup(f, "html.parser")
    except Exception:
        return ""

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

def preprocess(text: str) -> List[str]:
    """Lowercase, tokenize, remove stopwords, and lemmatize."""
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    tokens = [t for t in tokens if t not in stop_words]
    return [lemmatizer.lemmatize(t) for t in tokens]

def vectorize_text(text: str, vectorizer: TfidfVectorizer):
    """Transform text to a TF-IDF vector."""
    tokens = preprocess(text)
    return vectorizer.transform([" ".join(tokens)])
# endregion

# region INDEX CACHING
def cached_file_count() -> int:
    """Return previously stored number of indexed files."""
    if not os.path.exists(METADATA_FILE):
        return 0
    try:
        with open(METADATA_FILE, "r") as f:
            data = json.load(f)
        return data.get("file_count", 0)
    except Exception:
        return 0

def cache_file_count(count: int) -> None:
    """Store number of indexed files."""
    with open(METADATA_FILE, "w") as f:
        json.dump({"file_count": count}, f)

@st.cache_resource
def load_index(data_dir: str = "articles"):
    """Load or rebuild the TF-IDF index."""
    placeholder = st.empty()

    if not os.path.exists(data_dir):
        placeholder.text(f"Error: data directory '{data_dir}' not found.")
        return None, None, [], {}

    corpus_files = [
        f for f in os.listdir(data_dir)
        if f.lower().endswith((".html", ".htm"))
    ]

    rebuild = (
        not os.path.exists("tfidf_index.pkl")
        or len(corpus_files) != cached_file_count()
    )

    if not rebuild:
        placeholder.text("Loading cached index...")
        try:
            vectorizer, tfidf_matrix, doc_ids, doc_paths = joblib.load("tfidf_index.pkl")
            placeholder.text("Index loaded.")
            return vectorizer, tfidf_matrix, doc_ids, doc_paths
        except Exception:
            rebuild = True

    placeholder.text(f"Building TF-IDF index for {len(corpus_files)} documents...")

    corpus = {}
    for f in corpus_files:
        full_path = os.path.join(data_dir, f)
        corpus[os.path.splitext(f)[0]] = extract_text_from_html(full_path)

    corpus_tokens = {doc_id: preprocess(text) for doc_id, text in corpus.items()}
    docs_as_text = [" ".join(tokens) for tokens in corpus_tokens.values()]
    doc_ids = list(corpus_tokens.keys())

    vectorizer = TfidfVectorizer(max_features=10000, min_df=1)
    tfidf_matrix = vectorizer.fit_transform(docs_as_text)

    doc_paths = {
        os.path.splitext(f)[0]: os.path.join(data_dir, f)
        for f in corpus_files
    }

    joblib.dump((vectorizer, tfidf_matrix, doc_ids, doc_paths), "tfidf_index.pkl")
    cache_file_count(len(corpus_files))

    placeholder.text(f"Index built: ({len(doc_ids)} docs)")
    return vectorizer, tfidf_matrix, doc_ids, doc_paths
# endregion

# region QUERY EXPANSION
def expand_query(
    query: str,
    mode: int,
    vectorizer_vocab: Optional[Set[str]] = None,
    max_expansions: int = 5
) -> Tuple[str, Dict[str, Set[str]]]:
    """Expand, narrow, or preserve query based on user-selected mode."""

    tokens = preprocess(query)
    if not tokens:
        return "", {"original": set(), "core": set(), "expanded": set()}

    vectorizer_vocab = vectorizer_vocab or set()

    original_tokens = set(tokens)
    core_tokens: Set[str] = set()
    expanded_tokens: Set[str] = set()

    # mode: -1 Narrow
    if mode == -1:
        core_tokens = {w for w in tokens if wn.synsets(w, pos=wn.NOUN)}
        return " ".join(core_tokens), {
            "original": set(),
            "core": core_tokens,
            "expanded": set()
        }

    # mode: 0 Normal
    if mode == 0:
        core_tokens = set(tokens)
        return " ".join(tokens), {
            "original": original_tokens,
            "core": core_tokens,
            "expanded": set()
        }

    # mode: +1 Broad
    core_tokens = set(tokens)
    expanded = set(tokens)

    for word in tokens:
        word_exp = set()
        for pos in [wn.NOUN, wn.VERB, wn.ADJ]:
            for syn in wn.synsets(word, pos=pos):
                for lemma in syn.lemmas():
                    lemma_name = lemma.name().replace("_", " ").lower()
                    word_exp.add(lemmatizer.lemmatize(lemma_name))
                for hyper in syn.hypernyms():
                    for lemma in hyper.lemmas():
                        lemma_name = lemma.name().replace("_", " ").lower()
                        word_exp.add(lemmatizer.lemmatize(lemma_name))

        # keep vocabulary-compatible expansions
        if vectorizer_vocab:
            word_exp = {w for w in word_exp if w in vectorizer_vocab}

        limited = list(word_exp)[:max_expansions]
        expanded.update(limited)
        expanded_tokens.update(limited)

    weighted_query = tokens + tokens + list(expanded_tokens)
    return " ".join(weighted_query), {
        "original": original_tokens,
        "core": core_tokens,
        "expanded": expanded_tokens
    }
# endregion

# region SEARCH
def search(
    query: str,
    vectorizer: TfidfVectorizer,
    tfidf_matrix,
    doc_ids: List[str],
    doc_paths: Dict[str, str],
    top_k: int = 10
) -> List[Tuple[str, float]]:
    """Compute cosine similarity between query and documents."""
    if not query:
        return []

    try:
        query_vec = vectorize_text(query, vectorizer)
    except Exception:
        return []

    sims = linear_kernel(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]
    return [(doc_ids[i], sims[i]) for i in top_indices]
# endregion

# region STREAMLIT
st.title("Wikipedia Search Demo")

tab1, tab2 = st.tabs(["Search Results", "Project Overview"])

with tab1:
    if "show_main" not in st.session_state:
        st.session_state.show_main = False

    def show_main_callback():
        st.session_state.show_main = True

    if not st.session_state.show_main:
        st.write(
            "Welcome! If this is your first time here, building the index may take a minute or two. "
            "Once ready, you can search the articles immediately."
        )
        st.button("Continue", on_click=show_main_callback)

    else:
        vectorizer, tfidf_matrix, doc_ids, doc_paths = load_index()

        if vectorizer is None:
            st.error("Index failed to load. Please check your data folder.")
        else:
            query = st.text_input("Enter your query:")
            mode = st.slider("Query Expansion", -1, 1, 0, help="-1 narrow, 0 normal, 1 broad")

            if query.strip():
                expanded_query, token_types = expand_query(
                    query, mode,
                    vectorizer_vocab=set(vectorizer.vocabulary_.keys())
                )

                # displays color-coded query
                html_tokens = []
                for t in expanded_query.split():
                    if t in token_types["original"]:
                        color = COLOR_ORIGINAL
                    elif t in token_types["core"]:
                        color = COLOR_CORE
                    elif t in token_types["expanded"]:
                        color = COLOR_EXPANDED
                    else:
                        color = COLOR_OTHER

                    html_tokens.append(f"<span style='color:{color}'>{t}</span>")

                st.markdown("**Expanded query:**")
                st.markdown(" ".join(html_tokens), unsafe_allow_html=True)

                # execute search
                results = search(expanded_query, vectorizer, tfidf_matrix, doc_ids, doc_paths)

                st.subheader("Top Results:")
                for doc, score in results:
                    file_path = doc_paths.get(doc)
                    url = f"https://cwilburn-dev.github.io/INFO556Project/articles/{urllib.parse.quote(doc + '.htm')}"
                    if file_path and os.path.exists(file_path):
                        #link_html = f'<a href="{file_path}">{doc}</a>'
                        link_html = f'<a href="{url}" target="_blank" rel="noopener noreferrer">{doc}</a>'
                    else:
                        link_html = f"{doc} (file missing)"

                    st.markdown(
                        f'<div style="margin-bottom:0.25rem">{link_html} — {score:.3f}</div>',
                        unsafe_allow_html=True
                    )

with tab2:
    st.header("Project Overview")
    st.markdown(
        f"""
    **About**  
    This project implements a TF-IDF search engine over a small Wikipedia dataset.

    **Query Expansion Slider:**  
    - `-1` Narrow: only core terms  
    - `0` Normal: uses original preprocessed query  
    - `1` Broad: adds related terms from WordNet  

    **Color-Coded Terms:**  
    - <span style='color:{COLOR_ORIGINAL}'>Blue</span>: original query terms  
    - <span style='color:{COLOR_CORE}'>Red</span>: core/narrow terms  
    - <span style='color:{COLOR_EXPANDED}'>Green</span>: expanded terms  
    - <span style='color:{COLOR_OTHER}'>White</span>: other tokens  

    **Score Values:**  
    Higher scores indicate stronger query-document similarity.  

    **Suggested Searches:**  
    The dataset is fairly small, so not every query will return results.  Here are
    some suggested searchs to get you started:  
 
    - `architecture`  
    - `ocean liners of the 1900s`  
    - `methods of detecting online misinformation`  
    - `popular bands`  
    - `history of space exploration`  
    - `ocean predators`  
    """,
        unsafe_allow_html=True
    )
# endregion