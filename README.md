# content_based_filtering
An implementation of the tf*idf model for content-based filtering of the top 1000 IMDb movies

Input: imdb_top_1000.csv dataset from Kaggle, name of the film, number of recommendations for the given film
Output: a list of k recommendations for a given film
The csv file contains certain information on top 1000 films from IMDb. Given the purpose of the experiment, only the following columns were used: Series_title, 
Genre, Overview, Director, Star1, Star2, Star3, and Star4 as they provide the best information for content-based filtering. These columns were concatenated into 
a single column called text_data. This new column was then used as a corpus in the TF-IDF model, where a single entry in this column represents a single document/film. 
This approach is not perfect and would benefit from a more exhaustive text_data column (a movie plot instead of a short overview could be used instead to provide the best 
recommendations) or from a bigger list of films. However, this technique still provides adequate recommendations, especially for multi-chapter films or when k is low. 
A possible improvement to this model would be creating a list of movies the user likes and then getting the recommendation based on multiple films. 
This can be achieved using the same model by creating a new film list entry containing information about all liked items. As an example, if a user prefers two films, 
“The Godfather” and “The Dark Knight”, then a new entry to the dataset can be created by concatenating the titles, the overviews, and the names of the directors and actors in both movies.
This new entry could then be used as an input to the get_recommendation function.

-the main module contains the implementation of the tf*idf model using an already existing module sklearn.
-the visualizer module is a manual implementation of the tf*idf model
-content_based_filtering_manual is the module that imports the visualizer module to make a recommendation for a film.
