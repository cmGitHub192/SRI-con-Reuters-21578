import sys
import os
import re
from nltk.stem import PorterStemmer


### Carga de índice invertido
def load_inverted_index_from_txt(filepath):
    inverted_index = {}
    with open(filepath, 'r', encoding='utf-8') as file:
        current_term = None
        for line in file:
            line = line.strip()
            if line.startswith("Term:"):
                current_term = line.split("Term: ")[1]
                inverted_index[current_term] = []
            elif line.startswith("Document:"):
                doc_info = line.split("Document: ")[1]
                doc_name, weight = doc_info.split(", Weight: ")
                inverted_index[current_term].append((doc_name, float(weight)))
    return inverted_index


### Procesamiento

### Limpieza de texto
def clean_text(text):
    with open('stopwords', 'r', encoding='ascii') as file:
        stop_words = set(word.strip() for word in file.readlines())

    cleaned_text = re.sub(r'[^\w\s]', '', text)
    cleaned_text = cleaned_text.lower()
    tokens = cleaned_text.split()
    # Aplicar stemming
    stemmer = PorterStemmer()
    stemmed_tokens = [stemmer.stem(token) for token in tokens]
    # Eliminar stopwords
    cleaned_tokens = [token for token in stemmed_tokens if token not in stop_words]
    cleaned_text = ' '.join(cleaned_tokens)
    return cleaned_text

def process_query_previusly(query):
    cleaned_query = clean_text(query)
    return cleaned_query.split()

### Búsqueda
def jaccard_similarity(query_tokens, document_tokens):
    intersection = len(set(query_tokens) & set(document_tokens))
    union = len(set(query_tokens) | set(document_tokens))
    return intersection / union if union != 0 else 0  # Avoid division by zero

def search_jaccard(query, inverted_index):

    ### Carga de documentos
    CORPUS_DIR = 'training'
    documents = {}
    for filename in os.listdir(CORPUS_DIR):
        if filename.endswith(".txt"):
            filepath = os.path.join(CORPUS_DIR, filename)
            with open('stopwords', 'r', encoding='ascii') as file:
                text = file.read()
                cleaned_text = clean_text(text)
                documents[filename] = cleaned_text

    ### Busqueda propiamente
    query_tokens = process_query_previusly(query)
    document_tokens = {doc_id: documents[doc_id].split() for doc_id in documents}
    scores = {}
    for term in query_tokens:
        if term in inverted_index:
            for doc_id, tfidf_score in inverted_index[term]:
                if doc_id not in scores:
                    scores[doc_id] = 0
                scores[doc_id] += tfidf_score
    results = []
    for doc_id in scores:
        if doc_id in document_tokens:  # Verificar que la clave existe en document_tokens
            similarity = jaccard_similarity(query_tokens, document_tokens[doc_id])
            results.append((doc_id, similarity))
        else:
            print(f"Warning: Document {doc_id} not found in document_tokens")
    ranked_results = sorted(results, key=lambda x: x[1], reverse=True)
    
    # Extraer solo los doc_id de ranked_results
    doc_ids = [doc_id for doc_id, _ in ranked_results]
    
    return doc_ids

def results(query, tv, tr):
    if tv == "0":
        inverted_index_loaded = load_inverted_index_from_txt(os.path.join(os.getcwd(), 'results', 'inverted_index_tf_idf.txt'))
        if tr == "0":
            results = search_jaccard(query, inverted_index_loaded)
        else:
            results = search_jaccard(query, inverted_index_loaded)

    else:
        inverted_index_loaded = load_inverted_index_from_txt(os.path.join(os.getcwd(), 'results', 'inverted_index_bow.txt'))
        if tr == "0":
            results = search_jaccard(query, inverted_index_loaded)
        else:
            results = search_jaccard(query, inverted_index_loaded)
    # Búsqueda con Bag of Words
    return results

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Usage: python retrival.py <query> <tv> <tr>")
        sys.exit(1)

    query = sys.argv[1]
    tv = sys.argv[2] == '1'
    tr = sys.argv[3] == '1'
    result = results(query, tv, tr)
    print(result)