import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


class HybridRecommender:
    def __init__(self, movies, rating_sample):
        self.movies = movies.copy()
        self.ratings_sample = rating_sample.copy()
        self.content_similarity = None
        self.rating_similarity_df = None

    def fit(self):
        tfidf = TfidfVectorizer(stop_words="english")
        tfidf_matrix = tfidf.fit_transform(self.movies["content"])
        self.content_similarity = cosine_similarity(tfidf_matrix, tfidf_matrix)

        movie_user_matrix = self.ratings_sample.pivot_table(
            index="movieId",
            columns="userId",
            values="rating"
        )

        movie_user_matrix = movie_user_matrix.dropna(thresh=2, axis=0)
        movie_user_matrix = movie_user_matrix.fillna(0)

        rating_similarity_matrix = cosine_similarity(movie_user_matrix, movie_user_matrix)

        self.rating_similarity_df = pd.DataFrame(
            rating_similarity_matrix,
            index=movie_user_matrix.index,
            columns=movie_user_matrix.index
        )

    def recommend_movie(self, movie_title, top_n=5, content_weight=0.6, rating_weight=0.4):
        movie_match = self.movies[self.movies["title"].str.lower() == movie_title.lower()]

        if movie_match.empty:
            return f"Movie '{movie_title}' not found."

        movie_idx = movie_match.index[0]
        movie_id = movie_match.iloc[0]["movieId"]

        content_scores = list(enumerate(self.content_similarity[movie_idx]))
        content_scores_df = pd.DataFrame(content_scores, columns=["index", "content_score"])
        content_scores_df["movieId"] = self.movies.iloc[content_scores_df["index"]]["movieId"].values

        if movie_id in self.rating_similarity_df.index:
            rating_scores = self.rating_similarity_df.loc[movie_id].reset_index()
            rating_scores.columns = ["movieId", "rating_score"]
        else:
            rating_scores = pd.DataFrame({
                "movieId": self.movies["movieId"],
                "rating_score": 0.0
            })

        combined = pd.merge(
            content_scores_df[["movieId", "content_score"]],
            rating_scores,
            on="movieId",
            how="left"
        )

        combined["rating_score"] = combined["rating_score"].fillna(0)

        combined["final_score"] = (
            content_weight * combined["content_score"] +
            rating_weight * combined["rating_score"]
        )

        combined = combined[combined["movieId"] != movie_id]

        combined = pd.merge(
            combined,
            self.movies[["movieId", "title", "genres"]],
            on="movieId",
            how="left"
        )

        recommendations = combined.sort_values("final_score", ascending=False).head(top_n).copy()
        recommendations["content_score"] = recommendations["content_score"].round(3)
        recommendations["rating_score"] = recommendations["rating_score"].round(3)
        recommendations["final_score"] = recommendations["final_score"].round(3)

        return recommendations[["title", "genres", "content_score", "rating_score", "final_score"]]