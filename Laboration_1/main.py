from data import load_data
from preprocessing import prepare_movies, sample_ratings
from recommender import HybridRecommender

def main():
    zip_path = "/Users/wilgotlucaci/Desktop/ml-32m.zip"

    print("Loading data...")
    movies, ratings, tags = load_data(zip_path)

    print("Preparing movies...")
    movies = prepare_movies(movies, tags)

    print("Sampling ratings...")
    ratings_sample = sample_ratings(ratings, n=100_000)

    print("Building recommender...")
    recommender = HybridRecommender(movies, ratings_sample)
    recommender.fit()

    while True:
        movie_title = input("Enter a movie title: ")
        
        results = recommender.recommend_movie(movie_title)
        if  isinstance (results,str):
            print(results)
            print("Please try again.\n")
        else:
            print("\nRecommendations:")
            print(results)
            break
    
    

if __name__ == "__main__":
    main()