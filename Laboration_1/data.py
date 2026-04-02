import zipfile
import pandas as pd

def load_data(zip_path):
    with zipfile.ZipFile(zip_path) as z:
        movies = pd.read_csv(z.open("ml-32m/movies.csv"))
        ratings = pd.read_csv(z.open("ml-32m/ratings.csv"))
        tags = pd.read_csv(z.open("ml-32m/tags.csv"))
    return movies, ratings, tags