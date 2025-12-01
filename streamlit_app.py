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
import nltk

# region Setup
NLTK_DATA_DIR = os.path.join(os.getcwd(), "nltk_data")
os.makedirs(NLTK_DATA_DIR, exist_ok=True)
nltk.data.path.append(NLTK_DATA_DIR)
for pkg in ["stopwords", "wordnet", "omw-1.4"]:
    try:
        nltk.data.find(f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, download_dir=NLTK_DATA_DIR, quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()
# endregion

# region Preprocessing
def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)
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
    tokens = re.findall(r"\b[a-zA-Z]+\b", text.lower())
    tokens = [t for t in tokens if t not in stop_words]
    return [lemmatizer.lemmatize(t) for t in tokens]

def vectorize_text(text, vectorizer):
    tokens = preprocess(text)
    return vectorizer.transform([" ".join(tokens)])
# endregion

# region Index caching
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
    corpus = {os.path.splitext(f)[0]: extract_text_from_html(os.path.join(data_dir, f)) for f in corpus_files}
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
# endregion

# region Query expansion
def expand_query(query, mode, vectorizer_vocab=None, max_expansions=5):
    tokens = preprocess(query)
    if not tokens:
        return query, {"original": set(), "core": set(), "expanded": set()}

    if vectorizer_vocab is None:
        vectorizer_vocab = set()

    original_tokens = set(tokens)
    core_tokens = set()
    expanded_tokens = set()

    if mode == -1:
        # Narrow: only nouns
        core_tokens = {w for w in tokens if wn.synsets(w, pos=wn.NOUN)}
        return " ".join(core_tokens), {"original": set(), "core": core_tokens, "expanded": set()}

    if mode == 0:
        # Normal: full preprocessed query
        core_tokens = set(tokens)
        return " ".join(tokens), {"original": original_tokens, "core": core_tokens, "expanded": set()}

    # mode == 1 (broad): expand with WordNet
    core_tokens = set(tokens)
    expanded = set(tokens)

    for word in tokens:
        word_expansions = set()
        for pos in [wn.NOUN, wn.VERB, wn.ADJ]:
            for syn in wn.synsets(word, pos=pos):
                for lemma in syn.lemmas():
                    lemma_name = lemma.name().replace("_", " ").lower()
                    word_expansions.add(lemmatizer.lemmatize(lemma_name))
                for hyper in syn.hypernyms():
                    for lemma in hyper.lemmas():
                        lemma_name = lemma.name().replace("_", " ").lower()
                        word_expansions.add(lemmatizer.lemmatize(lemma_name))
        if vectorizer_vocab:
            word_expansions = {w for w in word_expansions if w in vectorizer_vocab}
        limited = list(word_expansions)[:max_expansions]
        expanded.update(limited)
        expanded_tokens.update(limited)

    weighted_query = tokens + tokens + list(expanded_tokens)
    return " ".join(weighted_query), {"original": original_tokens, "core": core_tokens, "expanded": expanded_tokens}
# endregion

# region Search
def search(query, vectorizer, tfidf_matrix, doc_ids, doc_paths, top_k=10):
    query_vec = vectorize_text(query, vectorizer)
    sims = linear_kernel(query_vec, tfidf_matrix).flatten()
    top_indices = np.argsort(sims)[::-1][:top_k]
    results = [(doc_ids[i], sims[i]) for i in top_indices]
    return results
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

        query = st.text_input("Enter your query:")
        mode = st.slider("Query Expansion", -1, 1, 0, help="-1 narrow, 0 normal, 1 broad")

        if query.strip():
            expanded_query, token_types = expand_query(query, mode)
            html_tokens = []
            for t in expanded_query.split():
                if t in token_types["original"]:
                    color = "#4DA6FF"  # blue
                elif t in token_types["core"]:
                    color = "#FF4D4D"  # red
                elif t in token_types["expanded"]:
                    color = "#33CC33"  # green
                else:
                    color = "#FFFFFF"

                html_tokens.append(f"<span style='color:{color}'>{t}</span>")

            st.markdown("**Expanded query:**")
            st.markdown(" ".join(html_tokens), unsafe_allow_html=True)

            results = search(expanded_query, vectorizer, tfidf_matrix, doc_ids, doc_paths)

            st.subheader("Top Results:")
            for doc, score in results:
                file_path = doc_paths.get(doc)
                if file_path and os.path.exists(file_path):
                    link_html = f'<a href="{file_path}">{doc}</a>'
                else:
                    link_html = doc + " (file missing)"
                st.markdown(
                    f'<div style="margin-bottom:0.25rem">{link_html} — {score:.3f}</div>',
                    unsafe_allow_html=True
                )

with tab2:
    st.header("Project Overview")
    st.markdown(f"""
    **About**  
    This project implements a TF-IDF search engine over a small Wikipedia dataset.

    **Query Expansion Slider:**  
    - `-1` Narrow: only core terms (short/important words)  
    - `0` Normal: uses original query (stopwords removed)  
    - `1` Broad: adds related terms from WordNet expansions  

    **Color-Coded Terms:**  
    - <span style='color:#4DA6FF'>Blue</span>: original query terms  
    - <span style='color:#FF4D4D'>Red</span>: core/narrow terms  
    - <span style='color:#33CC33'>Green</span>: expanded terms  
    - <span style='color:#FFFFFF'>White</span>: artifact from processing  

    **Score Values:**  
    - Higher scores indicate a better match between your query and the document  
    """, unsafe_allow_html=True)
# endregion