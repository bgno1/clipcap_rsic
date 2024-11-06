import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.util import ngrams


# CIDER for Chinese sentences
def compute_cider_for_image(gts, res, n=4):
    def compute_ngrams(text, n):
        tokens = text.split()
        ngram_list = []
        for i in range(1, n + 1):
            ngram_list.extend(list(ngrams(tokens, i)))
        return ngram_list

    def compute_tfidf_weighted_vectors(gts_ngrams, res_ngrams):
        tfidf_vectorizer = TfidfVectorizer(analyzer=lambda x: x)
        gts_ngrams_flattened = [' '.join([' '.join(ngram) for ngram in sum(gts_ngrams, [])])]
        res_ngrams_flattened = [' '.join([' '.join(ngram) for ngram in res_ngrams])]
        vectors = tfidf_vectorizer.fit_transform(gts_ngrams_flattened + res_ngrams_flattened)
        gts_vector = vectors[0].toarray()
        res_vector = vectors[1].toarray()
        return gts_vector, res_vector

    cider_scores = []

    for i in range(n):
        gts_ngrams = [compute_ngrams(gt, i + 1) for gt in gts]
        res_ngrams = compute_ngrams(res, i + 1)
        gts_vector, res_vector = compute_tfidf_weighted_vectors(gts_ngrams, res_ngrams)

        cosine_similarity = np.dot(gts_vector, res_vector.T) / (np.linalg.norm(gts_vector) * np.linalg.norm(res_vector))
        cider_scores.append(cosine_similarity[0][0])

    cider_score = np.mean(cider_scores)

    return cider_score


def compute_cider(gts_dict, res_dict, n=4):
    cider_scores = {}

    for image_id in gts_dict:
        gts = gts_dict[image_id]
        res = res_dict[image_id][0]
        cider_score = compute_cider_for_image(gts, res, n)
        cider_scores[image_id] = cider_score

    average_cider = np.mean(list(cider_scores.values()))

    return cider_scores, average_cider