# Wikipedia Search Demo
A lightweight search engine built with Streamlit that indexes locally stored Wikipedia HTML articles using TF-IDF.
The project demonstrates document preprocessing, vectorization, ranking, and adjustable query expansion using WordNet.

## Table of Contents

- [About](#about)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#structure)
- [Query Expansion](#query)
- [Requirements](#requirements)

---

## About
This project provides a simple text retrieval system designed for experimentation and instructional use.
Users can load a set of locally downloaded Wikipedia articles, build a TF-IDF index, and perform ranked retrieval through a Streamlit interface.

The system includes:

- HTML text extraction
- Tokenization, stopword filtering, and lemmatization
- TF-IDF vectorization
- Cosine similarity ranking
- WordNet-based query expansion

No external database is required; all data is processed locally.

## Features
- TF-IDF document indexing with scikit-learn
- Automatic index caching to improve performance
- Adjustable query expansion (narrow, neutral, broad)
- Extraction of paragraph text from Wikipedia HTML pages
- Streamlit interface for querying and displaying ranked results
- Works entirely offline once articles are downloaded

## Installation

### Clone Repository
git clone https://github.com/cwilburn-dev/INFO556Project.git  
cd INFO556Project

### Install Dependencies
The following libraries are required for the project:

- streamlit
- beautifulsoup4
- nltk
- scikit-learn
- numpy
- joblib

To install the dependencies, execute the command below:  
pip install -r requirements.txt

## Usage
Run the Streamlit application:  
streamlit run streamlit_app.py

Once running, the Streamlit app should open in your browser automatically.  

If not, navigate to:  
http://localhost:8501

---

## Query expansion
The query expansion slider supports three modes:
- −1 (Narrow): removes very short terms to tighten the query
- 0 (Neutral): searches using only the original query terms
- +1 (Broad): expands the query with WordNet synonyms and hypernyms (limited to selected semantic domains)

The expansion process includes token normalization and lemmatization.
