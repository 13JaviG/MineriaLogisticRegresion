from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import SelectKBest

# Convierte a formato tf_idf un conjunto de datos


def create_tfidf(data_frame):
    tfidf_vectorizer = TfidfVectorizer(use_idf=True)
    tfidf_matrix = tfidf_vectorizer.fit_transform(data_frame['open_response'].values.astype('U'))
    # instances = tfidf_matrix.A
    instances = tfidf_matrix.toarray()
    # Seleccionamos los k=600 mejores atributos, más significativos
    classes = list(data_frame['gs_text34'])
    instances = SelectKBest(k=700).fit_transform(instances, classes)
    return instances


