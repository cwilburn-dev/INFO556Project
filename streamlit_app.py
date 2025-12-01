# streamlit_app.py
import streamlit as st
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from gensim.models import KeyedVectors
from rank_bm25 import BM25Okapi
import joblib
import numpy as np

# ---------------------------
# NLTK setup
# ---------------------------
nltk.download("punkt", quiet=True)
nltk.download("averaged_perceptron_tagger", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("stopwords", quiet=True)

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

# ---------------------------
# Load precomputed index
# ---------------------------
@st.cache_resource
def load_index():
    bm25, doc_ids, doc_paths = joblib.load("files/bm25_index.pkl")
    doc_vectors = np.load("files/doc_vectors.npy")
    w2v_model = KeyedVectors.load("files/w2v_vectors.kv")
    return bm25, doc_ids, doc_paths, w2v_model, doc_vectors

# ---------------------------
# Query expansion
# ---------------------------
def contract_query(query_tokens, corpus_tokens):
    tokens = [lemmatizer.lemmatize(t) for t in query_tokens if t.isalpha()]
    tagged = nltk.pos_tag(tokens)
    nouns = [w for w, tag in tagged if tag.startswith("NN")]
    vocab = set().union(*corpus_tokens.values())
    nouns_in_corpus = [w for w in nouns if w in vocab]
    return nouns_in_corpus if nouns_in_corpus else nouns

def expand_query(query, mode, embedding_model=None, corpus_tokens=None, topn=3):
    tokens = [
        lemmatizer.lemmatize(t.lower())
        for t in nltk.word_tokenize(query)
        if t.isalpha() and t.lower() not in stop_words
    ]
    if not tokens:
        return query, {"original": [], "core": [], "expanded": []}

    original_tokens = tokens.copy()
    core_tokens = contract_query(tokens, corpus_tokens) if corpus_tokens else []

    bigrams = [f"{tokens[i]}_{tokens[i+1]}" for i in range(len(tokens)-1)]
    tokens_with_bigrams = tokens + bigrams

    expanded_tokens = set()
    if mode > 0 and embedding_model:
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
# Search & reranker
# ---------------------------
TOKEN_WEIGHTS = {"original": 4.0, "core": 3.0, "expanded": 0.6, "bigram": 2.0, "artifact": 0.0}
BONUS_MULT = 0.15
SEMANTIC_WEIGHT = 0.5
CANDIDATES_TO_RERANK = 50

def compute_query_vector(tokens, keyed_vectors):
    toks = [t for t in tokens if t in keyed_vectors.key_to_index]
    if not toks:
        return None
    vecs = [keyed_vectors.get_vector(t) for t in toks]
    mean_vec = np.mean(vecs, axis=0)
    norm = np.linalg.norm(mean_vec)
    return mean_vec / (norm + 1e-12) if norm else None

def search(expanded_query, bm25, doc_ids, corpus_tokens, doc_vectors=None, keyed_vectors=None, top_k=10):
    query_tokens = expanded_query.split()
    bm25_scores = np.array(bm25.get_scores(query_tokens))
    match_counts = [sum(1 for t in query_tokens if t in corpus_tokens[doc_id]) for doc_id in doc_ids]
    candidate_idxs = [i for i in range(len(doc_ids)) if match_counts[i] >= max(1, len(query_tokens)//4)]

    if not candidate_idxs:
        top_indices = np.argsort(bm25_scores)[::-1][:top_k]
        return [(doc_ids[i], float(bm25_scores[i])) for i in top_indices]

    top_candidates = sorted(candidate_idxs, key=lambda i: bm25_scores[i], reverse=True)[:CANDIDATES_TO_RERANK]
    qvec = compute_query_vector([t for t in query_tokens if "_" not in t], keyed_vectors) if keyed_vectors else None

    final_scores = []
    for i in top_candidates:
        doc_id = doc_ids[i]
        bm = float(bm25_scores[i])

        lexical_bonus = 0.0
        for t in query_tokens:
            if t in token_types.get("original", []) and t in corpus_tokens[doc_id]:
                lexical_bonus += TOKEN_WEIGHTS["original"]
            elif t in token_types.get("core", []) and t in corpus_tokens[doc_id]:
                lexical_bonus += TOKEN_WEIGHTS["core"]
            elif t in token_types.get("expanded", []) and t in corpus_tokens[doc_id]:
                lexical_bonus += TOKEN_WEIGHTS["expanded"]
            elif "_" in t and t in corpus_tokens[doc_id]:
                lexical_bonus += TOKEN_WEIGHTS["bigram"]
        lexical_bonus /= len(query_tokens) if query_tokens else 1
        score_after_lex = bm * (1.0 + BONUS_MULT * lexical_bonus)

        sem_score = float(np.dot(qvec, doc_vectors[i])) if qvec is not None else 0.0
        final_scores.append((i, score_after_lex + SEMANTIC_WEIGHT * sem_score))

    final_scores_sorted = sorted(final_scores, key=lambda x: x[1], reverse=True)[:top_k]
    return [(doc_ids[i], float(score)) for i, score in final_scores_sorted]

# ---------------------------
# Streamlit app
# ---------------------------
st.title("Wikipedia Search Demo")

tab1, tab2 = st.tabs(["Search Results", "Project Overview"])

with tab1:
    if "show_main" not in st.session_state:
        st.session_state.show_main = False

    if not st.session_state.show_main:
        st.write("Welcome! Index is precomputed and ready to use.")
        st.button("Continue", on_click=lambda: st.session_state.update({"show_main": True}))
    else:
        bm25, doc_ids, doc_paths, w2v_model, doc_vectors = load_index()
        corpus_tokens = {doc_id: [] for doc_id in doc_ids}  # minimal placeholder if not saved

        query = st.text_input("Enter your query:")
        mode = st.slider("Query Expansion", -1, 1, 0, help="-1 narrow, 0 normal, 1 broad")

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

            results = search(expanded_query, bm25, doc_ids, corpus_tokens, doc_vectors=doc_vectors, keyed_vectors=w2v_model)
            st.subheader("Top Results:")
            for doc, score in results:
                st.markdown(f"{doc} — {score:.3f}")

with tab2:
    st.header("Project Overview")
    st.markdown("""
**Query Expansion Slider:**  
- `-1` Narrow: core terms only  
- `0` Normal: original + bigrams  
- `1` Broad: adds related words from Word2Vec  

**Color-Coded Terms:**  
- Blue: original query terms  
- Red: core/narrow terms  
- Green: expanded terms  
- Grey: bigrams/derived tokens  
- White: fallback/other  

**Score Values:** Higher scores indicate better match.
""")
