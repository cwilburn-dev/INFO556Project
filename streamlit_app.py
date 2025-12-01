# app.py  (Streamlit Cloud friendly)
from bs4 import BeautifulSoup
import os
import re
import nltk
from nltk.data import find
from nltk.corpus import stopwords, wordnet as wn
from nltk.stem import WordNetLemmatizer
import numpy as np
import streamlit as st
import json
import time
from gensim.models import Word2Vec
from rank_bm25 import BM25Okapi

# ---------------------------
# NLTK (ensure downloads)
# ---------------------------
def ensure_nltk_data():
    resources = {
        'punkt': 'tokenizers/punkt',
        'stopwords': 'corpora/stopwords',
        'wordnet': 'corpora/wordnet',
        'omw-1.4': 'corpora/omw-1.4',
        'averaged_perceptron_tagger': 'taggers/averaged_perceptron_tagger',
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
    """
    Read an html file from the repo's articles/ directory and extract the main content.
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        soup = BeautifulSoup(f, 'html.parser')

    content_div = (
        soup.find('div', id='mw-content-text') or
        soup.find('div', id='bodyContent') or
        soup.find('body')
    )
    if not content_div:
        return ""

    paragraphs = content_div.find_all(['p', 'h1', 'h2', 'h3'])
    text = " ".join(p.get_text() for p in paragraphs)
    return clean_text(text)

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    # bigrams (underscored)
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
    return tokens + bigrams

# ---------------------------
# Indexing (in-memory, cached)
# ---------------------------
@st.cache_resource
def load_index(data_dir="articles"):
    """
    Build indexes in-memory and cache them for the session.
    Expects an 'articles/' folder in the app repo with .html files.
    Returns:
        bm25: BM25Okapi instance
        doc_ids: list of doc ids (filenames without extension)
        doc_paths: dict doc_id -> filepath
        w2v_wv: Word2Vec KeyedVectors (wv) object
        corpus_tokens: dict doc_id -> list(tokens+bigrams)
        doc_vectors: numpy array (n_docs, dim) of normalized mean vectors
        raw_texts: dict doc_id -> raw article text (for display)
    """
    placeholder = st.empty()
    placeholder.text("Building index (this runs once per session)...")

    # find html files in repo folder
    if not os.path.exists(data_dir):
        raise FileNotFoundError(f"Data directory '{data_dir}' not found in the repo. Add your article files there.")

    corpus_files = [f for f in os.listdir(data_dir) if f.lower().endswith((".html", ".htm"))]
    corpus = {}
    raw_texts = {}
    for f in corpus_files:
        doc_id = os.path.splitext(f)[0]
        path = os.path.join(data_dir, f)
        raw_html = open(path, 'r', encoding='utf-8').read()
        raw_texts[doc_id] = raw_html
        corpus[doc_id] = extract_text_from_html(path)

    # preprocess and token lists
    corpus_tokens = {doc_id: preprocess(text) for doc_id, text in corpus.items()}
    doc_ids = list(corpus_tokens.keys())
    corpus_token_lists = list(corpus_tokens.values())

    # Train Word2Vec on corpus token lists, keep it in memory (no saving to disk)
    # Use small workers to be safe on cloud; adjust vector_size as needed
    w2v_model = Word2Vec(sentences=corpus_token_lists, vector_size=100, window=5, min_count=1, workers=1)
    w2v_wv = w2v_model.wv

    # precompute document vectors (mean of token vectors) and normalize
    dim = w2v_wv.vector_size
    vectors = np.zeros((len(doc_ids), dim), dtype=np.float32)
    for i, doc_id in enumerate(doc_ids):
        toks = [t for t in corpus_tokens[doc_id] if t in w2v_wv.key_to_index and "_" not in t]
        if not toks:
            continue
        vecs = [w2v_wv.get_vector(t) for t in toks]
        mean_vec = np.mean(vecs, axis=0)
        norm = np.linalg.norm(mean_vec)
        if norm > 0:
            vectors[i] = mean_vec / (norm + 1e-12)

    # Build BM25
    bm25 = BM25Okapi(corpus_token_lists)

    # store doc_paths
    doc_paths = {doc_id: os.path.join(data_dir, f"{doc_id}.html") for doc_id in doc_ids}

    placeholder.text(f"Index built ✅ ({len(doc_ids)} docs)")
    time.sleep(0.6)
    placeholder.empty()

    return bm25, doc_ids, doc_paths, w2v_wv, corpus_tokens, vectors, raw_texts

# ---------------------------
# Query expansion
# ---------------------------
def contract_query(query_raw_tokens, corpus_tokens):
    tokens = [t for t in query_raw_tokens if t.isalpha()]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tagged = nltk.pos_tag(tokens)
    nouns = [w for w, tag in tagged if tag.startswith("NN")]
    vocab = set().union(*corpus_tokens.values()) if corpus_tokens else set()
    nouns_in_corpus = [w for w in nouns if w in vocab]
    core_terms = nouns_in_corpus if nouns_in_corpus else nouns
    return core_terms

def expand_query(query, mode, embedding_model=None, topn=3, corpus_tokens=None):
    tokens = [
        lemmatizer.lemmatize(t.lower())
        for t in nltk.word_tokenize(query)
        if t.isalpha() and t.lower() not in stop_words
    ]
    if not tokens:
        return "", {"original": [], "core": [], "expanded": []}

    original_tokens = tokens.copy()
    core_tokens = contract_query(tokens, corpus_tokens)

    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
    tokens_with_bigrams = tokens + bigrams

    expanded_tokens = set()
    if mode > 0 and embedding_model is not None:
        # expand from nouns and original tokens but filter low-similarity and non-nouns
        for word in tokens:
            if word in embedding_model.key_to_index:
                for sim_word, sim in embedding_model.most_similar(word, topn=topn):
                    pos_tag = nltk.pos_tag([sim_word])[0][1]
                    if sim > 0.55 and pos_tag.startswith("NN"):
                        expanded_tokens.add(sim_word)

    if mode == -1:
        weighted_query = core_tokens
    elif mode == 0:
        weighted_query = tokens_with_bigrams
    else:
        weighted_query = tokens_with_bigrams + list(expanded_tokens)

    token_types = {
        "original": original_tokens,
        "core": core_tokens if mode == -1 else [],
        "expanded": list(expanded_tokens) if mode > 0 else []
    }
    return " ".join(weighted_query), token_types

# ---------------------------
# Helpers: query/document vectors
# ---------------------------
def compute_query_vector(tokens, keyed_vectors):
    toks = [t for t in tokens if t in keyed_vectors.key_to_index and "_" not in t]
    if not toks:
        return None
    vecs = [keyed_vectors.get_vector(t) for t in toks]
    mean_vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm == 0:
        return None
    return mean_vec / (norm + 1e-12)

# ---------------------------
# Search & reranker
# ---------------------------
TOKEN_WEIGHTS = {
    "original": 4.0,
    "core": 3.0,
    "expanded": 0.6,
    "bigram": 2.0,
    "artifact": 0.0
}
BONUS_MULT = 0.15
SEMANTIC_WEIGHT = 0.5
CANDIDATES_TO_RERANK = 50

def search(expanded_query, token_types, bm25, doc_ids, corpus_tokens, doc_vectors=None, keyed_vectors=None,
           top_k=10, min_matches=None, min_score=None):
    # tokens as list (keep bigrams underscore form)
    query_tokens = [t for t in expanded_query.split() if t]
    bm25_scores = np.array(bm25.get_scores(query_tokens))

    if min_matches is None:
        min_matches = max(1, len(query_tokens) // 4)

    match_counts = [
        sum(1 for t in query_tokens if t in corpus_tokens[doc_id])
        for doc_id in doc_ids
    ]

    candidate_idxs = [
        i for i in range(len(doc_ids))
        if match_counts[i] >= min_matches and (min_score is None or bm25_scores[i] >= min_score)
    ]

    if len(candidate_idxs) == 0:
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        return [(doc_ids[i], float(bm25_scores[i])) for i in top_indices]

    candidate_idxs_sorted = sorted(candidate_idxs, key=lambda i: bm25_scores[i], reverse=True)
    top_candidates = candidate_idxs_sorted[:CANDIDATES_TO_RERANK]

    qvec = None
    if keyed_vectors is not None:
        qvec = compute_query_vector([t for t in query_tokens if "_" not in t], keyed_vectors)

    final_scores = []
    for i in top_candidates:
        doc_id = doc_ids[i]
        bm = float(bm25_scores[i])

        lexical_bonus = 0.0
        for t in query_tokens:
            if t in token_types.get("original", []):
                if t in corpus_tokens[doc_id]:
                    lexical_bonus += TOKEN_WEIGHTS["original"]
            elif t in token_types.get("core", []):
                if t in corpus_tokens[doc_id]:
                    lexical_bonus += TOKEN_WEIGHTS["core"]
            elif t in token_types.get("expanded", []):
                if t in corpus_tokens[doc_id]:
                    lexical_bonus += TOKEN_WEIGHTS["expanded"]
            elif "_" in t:
                if t in corpus_tokens[doc_id]:
                    lexical_bonus += TOKEN_WEIGHTS["bigram"]

        if len(query_tokens) > 0:
            lexical_bonus = lexical_bonus / len(query_tokens)

        score_after_lex = bm * (1.0 + BONUS_MULT * lexical_bonus)

        sem_score = 0.0
        if qvec is not None and doc_vectors is not None:
            doc_vec = doc_vectors[i]
            if np.linalg.norm(doc_vec) > 0:
                sem_score = float(np.dot(qvec, doc_vec))

        final = score_after_lex + SEMANTIC_WEIGHT * sem_score
        final_scores.append((i, final))

    final_scores_sorted = sorted(final_scores, key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_ids[i], float(score)) for i, score in final_scores_sorted]

# ---------------------------
# Streamlit UI
# ---------------------------
st.set_page_config(page_title="Query Expansion Demo", layout="wide")
st.title("Query Expansion — Interactive Demo")

tab1, tab2 = st.tabs(["Search Results", "Project Overview"])

with tab1:
    if "show_main" not in st.session_state:
        st.session_state.show_main = False

    def show_main_callback():
        st.session_state.show_main = True

    if not st.session_state.show_main:
        st.write(
            "Welcome! Building index may take a little time the first session. "
            "Once ready, you can search the articles immediately."
        )
        st.button("Continue", on_click=show_main_callback)
    else:
        # load index (cached in memory)
        try:
            bm25, doc_ids, doc_paths, w2v_wv, corpus_tokens, doc_vectors, raw_texts = load_index()
        except FileNotFoundError as e:
            st.error(str(e))
            st.stop()

        cols = st.columns([3, 1])
        with cols[0]:
            query = st.text_input("Enter your query:")
        with cols[1]:
            mode = st.slider("Query Expansion", -1, 1, 0, help="-1 narrow, 0 normal, 1 broad")

        if query is not None and query.strip():
            expanded_query, token_types = expand_query(
                query, mode, embedding_model=w2v_wv, corpus_tokens=corpus_tokens
            )

            # Show colored tokens
            html_tokens = []
            for t in nltk.word_tokenize(expanded_query):
                if t in token_types["core"]:
                    color = "#FF4D4D"
                elif t in token_types["expanded"]:
                    color = "#33CC33"
                elif t in token_types["original"]:
                    color = "#4DA6FF"
                elif "_" in t:
                    color = "#AAAAAA"
                else:
                    color = "#FFFFFF"
                html_tokens.append(f"<span style='color:{color}'>{t}</span>")

            st.markdown("**Expanded query:**")
            st.markdown(" ".join(html_tokens), unsafe_allow_html=True)

            # Run search
            results = search(expanded_query, token_types, bm25, doc_ids, corpus_tokens,
                             doc_vectors=doc_vectors, keyed_vectors=w2v_wv,
                             top_k=10, min_matches=1, min_score=None)

            st.subheader("Top Results:")
            for doc, score in results:
                # doc is doc_id (filename without ext)
                st.markdown(f"**{doc}** — {score:.3f}")
                # show snippet and allow download of original HTML
                raw_html = raw_texts.get(doc, "")
                snippet = BeautifulSoup(raw_html, "html.parser").get_text()[:800]
                with st.expander("Show excerpt & download HTML"):
                    st.write(snippet + ("..." if len(snippet) < len(raw_html) else ""))
                    # provide a download button with the HTML content
                    st.download_button(label="Download HTML", data=raw_html, file_name=f"{doc}.html", mime="text/html")

with tab2:
    st.header("Project Overview")
    st.markdown("""
    **About**  
    This project exposes query expansion as a user-controlled feature.

    **Slider modes:**  
    - `-1` Narrow: contract query to core nouns  
    - `0` Normal: original query (stopwords removed + bigrams)  
    - `1` Broad: add related nouns from Word2Vec

    **Color key:**  
    - Blue: original terms  
    - Red: core terms (when contracted)  
    - Green: expanded terms  
    - Grey: bigrams / derived tokens
    """, unsafe_allow_html=True)
