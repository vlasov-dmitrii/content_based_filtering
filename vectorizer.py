# Authors: Dmitrii Vlasov, Armin Pousti
import math
import numpy as np

# example
docs = [
    "This is the first document this document.",
    "This is the second document.",
    "And this is the third one.",
    "Is this the first document first document ?"
]


def preprocess(text):
    """
    Preprocesses the given text by converting it to lowercase, splitting it into words,
    and removing common stopwords.
    :param text: (str) text to preprocess
    :return: (list) preprocessed list of words
    """

    # convert the string to lower case
    text = text.lower()

    # split into individual words
    words = text.split()

    # remove common stopwords
    stopwords = ['a', 'an', 'and', 'are', 'as', 'at', 'be', 'by', 'for', 'from', 'has', 'he', 'in', 'is', 'it', 'its', 'of', 'on', 'that', 'the', 'to', 'was', 'with', 'you']
    new_words = []
    for word in words:
        if word not in stopwords:
            new_words.append(word)
    words = new_words

    return words


def compute_tf(words):
    """
    Computes the term frequency (TF) for each word in the document
    :param words: (list) the list of words in the document
    :return: (dict) the term frequency dictionary
    """
    # tf - term frequency
    # count the frequency of each word in the text
    tf_dict = {}
    for word in words:
        # increment the value of the key "word" by 1
        if word in tf_dict:
            tf_dict[word] += 1
        else:
            tf_dict[word] = 1

    # divide by the total number of words to obtain the word frequencies
    for word in tf_dict:
        tf_dict[word] = tf_dict[word] / len(words)

    return tf_dict


def compute_idf(tokenized_docs):
    """
    Computes the inverse document frequency for each term
    :param tokenized_docs: (list) the list of tokenized documents
    :return: (dict) the inverse document frequency dictionary
    """
    # idf - inverse document frequency
    doc_freq_dict = {}
    for tokens in tokenized_docs:
        unique_tokens = set(tokens)
        for token in unique_tokens:
            if token in doc_freq_dict:
                doc_freq_dict[token] += 1
            else:
                doc_freq_dict[token] = 1

    # compute the inverse document frequency
    num_docs = len(tokenized_docs)
    idf_dict = {}
    for token in doc_freq_dict:
        idf_dict[token] = math.log(num_docs / (doc_freq_dict[token]))
    return idf_dict


def compute_tfidf(tf, idf):
    """
    Computes the TF-IDF values for each term in the document
    :param tf: (dict) term frequency dictionary
    :param idf: (dict) inverse document frequency dictionary
    :return: (dict) TF-IDF dictionary
    """
    # compute the TF-IDF values for each token in the document
    tfidf_dict = {}
    for token in tf:
        tfidf_dict[token] = tf[token] * idf[token]

    return tfidf_dict

def compute_cosine_similarity(docs):
    """
    Computes cosine similarity matrix for a list of documents
    :param docs: (list) the list of documents
    :return: (numpy array) cosine similarity matrix
    """
    # preprocess the documents
    tokenized_docs = []
    for doc in docs:
        tokenized_doc = preprocess(doc)
        tokenized_docs.append(tokenized_doc)
    # compute the TF-IDF vectors for each document
    tfidf_vectors = []
    idf = compute_idf(tokenized_docs)
    for tokenized_doc in tokenized_docs:
        tf = compute_tf(tokenized_doc)
        tfidf = compute_tfidf(tf, idf)
        tfidf_vectors.append(tfidf)

    # calculate the cosine similarity matrix
    num_docs = len(docs)
    cosine_similarity_matrix = np.zeros((num_docs, num_docs))
    for i in range(num_docs):
        for j in range(num_docs):
            if i != j:
                doc1_tfidf = tfidf_vectors[i]
                doc2_tfidf = tfidf_vectors[j]

                # dot product
                common_tokens = set(doc1_tfidf) & set(doc2_tfidf)
                dot_product = 0
                for token in common_tokens:
                    tfidf1 = doc1_tfidf.get(token, 0)
                    tfidf2 = doc2_tfidf.get(token, 0)
                    # calculate the product of the TF-IDF values and add it to the dot product
                    dot_product += tfidf1 * tfidf2
                norm_doc1 = np.linalg.norm(list(doc1_tfidf.values()))
                norm_doc2 = np.linalg.norm(list(doc2_tfidf.values()))
                cosine_similarity_matrix[i, j] = dot_product / (norm_doc1 * norm_doc2)

    return cosine_similarity_matrix

def get_recommendation(cos_sim_matrix, k, index):
    """
    Gets the best k recommendations based on the cosine similarity matrix for a given index
    :param cos_sim_matrix: (numpy array) cosine similarity matrix
    :param k: (int) the number of the best recommendations
    :param index: (int) index of the document for recommendations
    :return: (numpy array) indices of the best k recommendations
    """
    # sort the similarity scores in descending order and get the indices of the top k similar documents for recommendation
    top_k_indices = np.argsort(cos_sim_matrix[index])[::-1][:k]
    return top_k_indices
