import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from bs4 import BeautifulSoup
import re
import numpy as np
from rank_bm25 import BM25Okapi
from gensim.models import KeyedVectors
import requests
from io import BytesIO

# ---------------------------
# NLTK setup
# ---------------------------
nltk.download("punkt", quiet=True)
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------------------
# Preprocessing
# ---------------------------
def clean_text(text):
    text = re.sub(r"\[\d+\]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def preprocess(text):
    tokens = nltk.word_tokenize(text.lower())
    tokens = [t for t in tokens if t.isalpha() and t not in stop_words]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
    return tokens + bigrams

def extract_text_from_html_string(html_string):
    soup = BeautifulSoup(html_string, "html.parser")
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

# ---------------------------
# Load precomputed data
# ---------------------------
@st.cache_resource
def load_precomputed_data():
    # Load BM25 corpus (assumes articles hosted on GitHub)
    # Example: {'doc_id': 'https://raw.githubusercontent.com/.../article.html'}
    corpus_urls = {
        "Python": "https://raw.githubusercontent.com/<your_repo>/articles/python.html",
        "NLP": "https://raw.githubusercontent.com/<your_repo>/articles/nlp.html",
        # add all your documents here
    }

    corpus_tokens = {}
    doc_html = {}

    for doc_id, url in corpus_urls.items():
        r = requests.get(url)
        html_content = r.text
        doc_html[doc_id] = html_content
        text = extract_text_from_html_string(html_content)
        corpus_tokens[doc_id] = preprocess(text)

    bm25 = BM25Okapi(list(corpus_tokens.values()))

    # Load precomputed Word2Vec KeyedVectors from GitHub
    kv_url = "https://raw.githubusercontent.com/<your_repo>/w2v_vectors.kv"
    r = requests.get(kv_url)
    w2v_model = KeyedVectors.load(BytesIO(r.content), mmap="r")

    # Precompute doc vectors
    doc_vectors = compute_doc_vectors(corpus_tokens, w2v_model)

    return bm25, list(corpus_tokens.keys()), corpus_tokens, w2v_model, doc_vectors, doc_html

def compute_doc_vectors(corpus_tokens, keyed_vectors):
    dim = keyed_vectors.vector_size
    doc_ids = list(corpus_tokens.keys())
    vectors = np.zeros((len(doc_ids), dim), dtype=np.float32)
    for i, doc_id in enumerate(doc_ids):
        toks = [t for t in corpus_tokens[doc_id] if t in keyed_vectors.key_to_index]
        if toks:
            vecs = [keyed_vectors.get_vector(t) for t in toks]
            mean_vec = np.mean(vecs, axis=0)
            vectors[i] = mean_vec / (np.linalg.norm(mean_vec) + 1e-12)
    return vectors

def compute_query_vector(tokens, keyed_vectors):
    toks = [t for t in tokens if t in keyed_vectors.key_to_index]
    if not toks:
        return None
    vecs = [keyed_vectors.get_vector(t) for t in toks]
    mean_vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean_vec)
    if norm == 0:
        return None
    return mean_vec / (norm + 1e-12)

# ---------------------------
# Query expansion
# ---------------------------
def contract_query(query_raw_tokens, corpus_tokens):
    tokens = [t for t in query_raw_tokens if t.isalpha()]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tagged = nltk.pos_tag(tokens)
    nouns = [w for w, tag in tagged if tag.startswith("NN")]
    vocab = set().union(*corpus_tokens.values())
    nouns_in_corpus = [w for w in nouns if w in vocab]
    return nouns_in_corpus if nouns_in_corpus else nouns

def expand_query(query, mode, embedding_model=None, topn=3, corpus_tokens=None):
    tokens = [
        lemmatizer.lemmatize(t.lower())
        for t in nltk.word_tokenize(query)
        if t.isalpha() and t.lower() not in stop_words
    ]
    if not tokens:
        return query, {"original": [], "core": [], "expanded": []}
    original_tokens = tokens.copy()
    core_tokens = contract_query(tokens, corpus_tokens)
    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
    tokens_with_bigrams = tokens + bigrams
    expanded_tokens = set()
    if mode > 0 and embedding_model is not None:
        for word in tokens:
            if word in embedding_model.key_to_index:
                for sim_word, sim in embedding_model.most_similar(word, topn=topn):
                    pos_tag = nltk.pos_tag([sim_word])[0][1]
                    if sim > 0.6 and pos_tag.startswith("NN"):
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
# Search / rerank
# ---------------------------
TOKEN_WEIGHTS = {"original":4.0, "core":3.0, "expanded":0.6, "bigram":2.0, "artifact":0.0}
BONUS_MULT = 0.15
SEMANTIC_WEIGHT = 0.5
CANDIDATES_TO_RERANK = 50

def search(expanded_query, bm25, doc_ids, corpus_tokens, doc_vectors=None, keyed_vectors=None,
           top_k=10, min_matches=None, min_score=None, token_types=None):
    query_tokens = expanded_query.split()
    bm25_scores = np.array(bm25.get_scores(query_tokens))
    if min_matches is None:
        min_matches = max(1, len(query_tokens)//4)
    match_counts = [sum(1 for t in query_tokens if t in corpus_tokens[doc_id]) for doc_id in doc_ids]
    candidate_idxs = [
        i for i in range(len(doc_ids))
        if match_counts[i]>=min_matches and (min_score is None or bm25_scores[i]>=min_score)
    ]
    if len(candidate_idxs)==0:
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        return [(doc_ids[i], float(bm25_scores[i])) for i in top_indices]
    top_candidates = sorted(candidate_idxs, key=lambda i: bm25_scores[i], reverse=True)[:CANDIDATES_TO_RERANK]
    qvec = None
    if keyed_vectors is not None:
        qvec = compute_query_vector([t for t in query_tokens if "_" not in t], keyed_vectors)
    final_scores = []
    for i in top_candidates:
        doc_id = doc_ids[i]
        bm = float(bm25_scores[i])
        lexical_bonus = 0.0
        for t in query_tokens:
            if token_types:
                if t in token_types.get("original",[]):
                    if t in corpus_tokens[doc_id]:
                        lexical_bonus += TOKEN_WEIGHTS["original"]
                elif t in token_types.get("core",[]):
                    if t in corpus_tokens[doc_id]:
                        lexical_bonus += TOKEN_WEIGHTS["core"]
                elif t in token_types.get("expanded",[]):
                    if t in corpus_tokens[doc_id]:
                        lexical_bonus += TOKEN_WEIGHTS["expanded"]
                elif "_" in t and t in corpus_tokens[doc_id]:
                    lexical_bonus += TOKEN_WEIGHTS["bigram"]
        if len(query_tokens)>0:
            lexical_bonus /= len(query_tokens)
        score_after_lex = bm*(1.0+BONUS_MULT*lexical_bonus)
        sem_score = 0.0
        if qvec is not None and doc_vectors is not None:
            doc_vec = doc_vectors[i]
            if np.linalg.norm(doc_vec)>0:
                sem_score = float(np.dot(qvec, doc_vec))
        final = score_after_lex + SEMANTIC_WEIGHT*sem_score
        final_scores.append((i, final))
    final_scores_sorted = sorted(final_scores, key=lambda x:x[1], reverse=True)[:top_k]
    return [(doc_ids[i], float(score)) for i, score in final_scores_sorted]

# ---------------------------
# Streamlit UI
# ---------------------------
st.title("Wikipedia Search Demo (Streamlit Cloud)")

tab1, tab2 = st.tabs(["Search Results","Project Overview"])

with tab1:
    bm25, doc_ids, corpus_tokens, w2v_model, doc_vectors, doc_html = load_precomputed_data()

    query = st.text_input("Enter your query:")
    mode = st.slider("Query Expansion", -1, 1, 0,
                     help="-1 narrow, 0 normal, 1 broad")

    if query.strip():
        expanded_query, token_types = expand_query(query, mode, embedding_model=w2v_model, corpus_tokens=corpus_tokens)

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

        results = search(expanded_query, bm25, doc_ids, corpus_tokens, doc_vectors=doc_vectors,
                         keyed_vectors=w2v_model, token_types=token_types, top_k=10, min_matches=1, min_score=0)

        st.subheader("Top Results:")
        for doc, score in results:
            url = doc_html.get(doc, "#")
            st.markdown(f'<div style="margin-bottom:0.25rem"><a href="{url}">{doc}</a> — {score:.3f}</div>', unsafe_allow_html=True)

with tab2:
    st.header("Project Overview")
    st.markdown("""
    **About**  
    This project moves query expansion from a hidden backend process to an interactive, user-controlled feature.

    **Query Expansion Slider:**  
    - `-1` Narrow: only core terms (nouns)  
    - `0` Normal: original query  
    - `1` Broad: adds related words from Word2Vec  

    **Color-Coded Terms:**  
    - Blue: original query  
    - Red: core terms  
    - Green: expanded terms  
    - Grey: bigrams  
    """, unsafe_allow_html=True)
