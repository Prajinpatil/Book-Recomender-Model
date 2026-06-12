"""
recommender.py
--------------
Content-based book recommendation engine using TF-IDF vectorization
and cosine similarity. Designed for RFID-integrated library systems.
"""

import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from pathlib import Path


class BookRecommender:
    """
    Content-based filtering recommender that generates personalized,
    topic-aware book recommendations from a user's borrowing history.

    Pipeline:
        Load data → Build feature strings → TF-IDF vectorize →
        Compute cosine similarity → Rank and filter → Return Top-N
    """

    def __init__(self, books_path: str, history_path: str):
        self.books_path = Path(books_path)
        self.history_path = Path(history_path)

        self.books_df: pd.DataFrame = None
        self.history_df: pd.DataFrame = None
        self.tfidf_matrix = None
        self.similarity_matrix = None
        self.vectorizer = TfidfVectorizer(stop_words="english")

        self._load_data()
        self._build_tfidf_matrix()

    # ------------------------------------------------------------------ #
    #  Data Loading                                                        #
    # ------------------------------------------------------------------ #

    def _load_data(self) -> None:
        """Load books catalogue and borrowing history from CSV files."""
        self.books_df = pd.read_csv(self.books_path)
        self.history_df = pd.read_csv(self.history_path)

        # Validate required columns
        required_books = {"book_id", "title", "author", "category", "tags"}
        required_history = {"user_id", "book_id"}

        if not required_books.issubset(self.books_df.columns):
            raise ValueError(f"books.csv must contain columns: {required_books}")
        if not required_history.issubset(self.history_df.columns):
            raise ValueError(f"borrowing_history.csv must contain columns: {required_history}")

    # ------------------------------------------------------------------ #
    #  Feature Engineering                                                 #
    # ------------------------------------------------------------------ #

    def _build_feature_string(self, row: pd.Series) -> str:
        """
        Combine book metadata into a single text string for vectorization.

        Format: "<title> <category> <tags>"
        """
        parts = [
            str(row.get("title", "")),
            str(row.get("author", "")),
            str(row.get("category", "")),
            str(row.get("tags", "")),
        ]
        return " ".join(filter(None, parts)).lower()

    def _build_tfidf_matrix(self) -> None:
        """Vectorize all book feature strings and compute pairwise similarity."""
        self.books_df["feature_string"] = self.books_df.apply(
            self._build_feature_string, axis=1
        )
        self.tfidf_matrix = self.vectorizer.fit_transform(
            self.books_df["feature_string"]
        )
        self.similarity_matrix = cosine_similarity(self.tfidf_matrix)

    # ------------------------------------------------------------------ #
    #  Core Recommendation Logic                                           #
    # ------------------------------------------------------------------ #

    def get_similar_books(self, book_id: str, top_n: int = 5) -> pd.DataFrame:
        """
        Return the top-N most similar books to a given book ID.

        Args:
            book_id: The ID of the reference book (e.g., 'B001').
            top_n:   Number of similar books to return.

        Returns:
            DataFrame with columns [book_id, title, author, category, similarity_score].
        """
        if book_id not in self.books_df["book_id"].values:
            raise ValueError(f"Book ID '{book_id}' not found in catalogue.")

        idx = self.books_df.index[self.books_df["book_id"] == book_id].item()
        sim_scores = list(enumerate(self.similarity_matrix[idx]))

        # Sort by similarity descending, exclude the book itself
        sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)
        sim_scores = [(i, s) for i, s in sim_scores if i != idx][:top_n]

        result_indices = [i for i, _ in sim_scores]
        result_scores = [round(s, 4) for _, s in sim_scores]

        result = self.books_df.iloc[result_indices][
            ["book_id", "title", "author", "category"]
        ].copy()
        result["similarity_score"] = result_scores
        return result.reset_index(drop=True)

    def recommend_for_user(self, user_id: str, top_n: int = 5) -> pd.DataFrame:
        """
        Generate personalized recommendations based on a user's borrowing history.

        Steps:
            1. Retrieve all books the user has borrowed.
            2. Aggregate similarity scores across all borrowed books.
            3. Remove already-read books.
            4. Return top-N ranked recommendations.

        Args:
            user_id: The RFID-linked user ID (e.g., 'U001').
            top_n:   Number of recommendations to return.

        Returns:
            DataFrame with columns [book_id, title, author, category, aggregated_score].
        """
        user_history = self.history_df[self.history_df["user_id"] == user_id]

        if user_history.empty:
            raise ValueError(f"No borrowing history found for user '{user_id}'.")

        borrowed_ids = set(user_history["book_id"].values)
        score_accumulator: dict[str, float] = {}

        for book_id in borrowed_ids:
            if book_id not in self.books_df["book_id"].values:
                continue  # skip books not in catalogue

            idx = self.books_df.index[self.books_df["book_id"] == book_id].item()
            for i, score in enumerate(self.similarity_matrix[idx]):
                candidate_id = self.books_df.iloc[i]["book_id"]
                if candidate_id not in borrowed_ids:
                    score_accumulator[candidate_id] = (
                        score_accumulator.get(candidate_id, 0.0) + score
                    )

        if not score_accumulator:
            return pd.DataFrame(
                columns=["book_id", "title", "author", "category", "aggregated_score"]
            )

        ranked = sorted(score_accumulator.items(), key=lambda x: x[1], reverse=True)[:top_n]
        rec_ids = [bid for bid, _ in ranked]
        rec_scores = [round(s, 4) for _, s in ranked]

        result = self.books_df[self.books_df["book_id"].isin(rec_ids)][
            ["book_id", "title", "author", "category"]
        ].copy()

        # Preserve ranking order
        score_map = dict(zip(rec_ids, rec_scores))
        result["aggregated_score"] = result["book_id"].map(score_map)
        result = result.sort_values("aggregated_score", ascending=False)

        return result.reset_index(drop=True)

    # ------------------------------------------------------------------ #
    #  Utility                                                             #
    # ------------------------------------------------------------------ #

    def get_user_history(self, user_id: str) -> pd.DataFrame:
        """Return the borrowing history for a user with full book details."""
        user_history = self.history_df[self.history_df["user_id"] == user_id]
        if user_history.empty:
            raise ValueError(f"No history found for user '{user_id}'.")
        return user_history.merge(self.books_df, on="book_id", how="left")

    def catalogue_size(self) -> int:
        """Return total number of books in the catalogue."""
        return len(self.books_df)
