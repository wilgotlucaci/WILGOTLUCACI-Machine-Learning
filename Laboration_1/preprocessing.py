import pandas as pd

def prepare_movies(movies, tags):
    tag_data = tags.groupby("movieId")["tag"].apply(lambda x: " ".join(x.dropna().astype(str))).reset_index()

    movies = pd.merge(movies, tag_data, on="movieId", how="left")
    movies["tag"] = movies["tag"].fillna("")
    movies["genres"] = movies["genres"].str.replace("|", " ", regex=False)
    movies["content"] = movies["genres"] + " " + movies["tag"]
    return movies

def sample_ratings(ratings, n=100_000, random_state= 42):
    return ratings.sample(n=n, random_state=random_state)