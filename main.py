# Authors: Dmitrii Vlasov, Armin Pousti
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read the CSV file into a DataFrame
df = pd.read_csv('imdb_top_1000.csv')

# Access the data in the DataFrame

# removing unnecessary columns
# define a list of columns to remove
columns_to_remove = ['Poster_Link', 'Certificate', 'No_of_Votes', 'Gross']
# remove the specified columns
df = df.drop(columns_to_remove, axis=1)

# concatenate relevant text-based columns
df['text_data'] = df['Series_Title'] + ' ' + df['Overview'] + ' ' + df['Genre'] + ' ' + df['Director'] + ' ' + df['Star1'] + ' ' + df['Star2'] + ' ' + df['Star3'] + ' ' + df['Star4']
# initialize the TF-IDF vectorizer
tfidf_vectorizer = TfidfVectorizer()

# apply TF-IDF vectorization on the text data
tfidf_matrix = tfidf_vectorizer.fit_transform(df['text_data'])
# get the feature names (words) in the TF-IDF matrix
feature_names = tfidf_vectorizer.get_feature_names_out()

# calculate cosine similarity matrix
cosine_sim_matrix = cosine_similarity(tfidf_matrix, tfidf_matrix)

# given film of interest
given_film_title = 'Fight Club'  # Replace with the film you want recommendations for

# find the index of the given film in the DataFrame
given_film_index = df[df['Series_Title'] == given_film_title].index[0]

# get the similarity scores of the given film with all other films
similarity_scores = cosine_sim_matrix[given_film_index]

# sort the similarity scores in descending order and get the indices
sorted_indices = similarity_scores.argsort()[::-1]

recommended_indices = sorted_indices[1:10]
recommended_films = df.iloc[recommended_indices]['Series_Title'].values
print(recommended_films)