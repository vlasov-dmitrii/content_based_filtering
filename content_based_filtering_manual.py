# Authors: Dmitrii Vlasov, Armin Pousti
import pandas as pd
import vectorizer

df = pd.read_csv('imdb_top_1000.csv')

columns_to_remove = ['Poster_Link', 'Certificate', 'No_of_Votes', 'Gross']
# Remove the specified columns
df = df.drop(columns_to_remove, axis=1)
df = df[0:100]
# Concatenate relevant text-based columns
df['text_data'] = df['Series_Title'] + ' ' + df['Overview'] + ' ' + df['Genre'] + ' ' + df['Director'] + ' ' + df['Star1'] + ' ' + df['Star2'] + ' ' + df['Star3'] + ' ' + df['Star4']
# Iterate over rows of df
docs = []
for index, row in df.iterrows():
    docs.append(row['text_data'])


def recommendations(k, docs, name):
    """
    Gives k best recommendations for a given film in docs
    :param k: (int) number of the best recommendations
    :param docs: (list) the list of documents
    :param name: (str) name of the film for which the recommendations are needed
    :return: (list) list of k best films that are similar to a given film
    """
    matrix = vectorizer.compute_cosine_similarity(docs)
    for index, row in df.iterrows():
        if name == row['Series_Title']:
            i = index;
    recommended_indeces = vectorizer.get_recommendation(matrix, k, i)

    recommended_list = []
    for index in recommended_indeces:
        recommended_list.append(df.loc[index]['Series_Title'])
    return recommended_list

# get 3 best recommendations for the film "Fight Club"
print(recommendations(3, docs, "Fight Club"))



