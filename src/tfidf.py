from sklearn.feature_extraction.text import TfidfVectorizer


# Convierte a formato tf_idf un conjunto de datos

def create_tfidf(data_frame):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_frame['open_response'].values.astype('U'))

    return tfidf_matrix.A

