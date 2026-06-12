"""
tests/test_recommender.py
--------------------------
Unit tests for the BookRecommender engine.
Run with: pytest tests/ -v
"""

import sys
from pathlib import Path
import pytest
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
from src.recommender import BookRecommender

BOOKS_PATH = Path(__file__).parent.parent / "data" / "books.csv"
HISTORY_PATH = Path(__file__).parent.parent / "data" / "borrowing_history.csv"


@pytest.fixture(scope="module")
def recommender():
    return BookRecommender(str(BOOKS_PATH), str(HISTORY_PATH))


class TestDataLoading:
    def test_books_loaded(self, recommender):
        assert recommender.catalogue_size() > 0

    def test_required_columns_exist(self, recommender):
        for col in ["book_id", "title", "author", "category", "tags"]:
            assert col in recommender.books_df.columns

    def test_tfidf_matrix_shape(self, recommender):
        n = recommender.catalogue_size()
        assert recommender.similarity_matrix.shape == (n, n)


class TestSimilarBooks:
    def test_returns_dataframe(self, recommender):
        result = recommender.get_similar_books("B001", top_n=5)
        assert isinstance(result, pd.DataFrame)

    def test_correct_number_of_results(self, recommender):
        result = recommender.get_similar_books("B001", top_n=3)
        assert len(result) == 3

    def test_source_book_not_in_results(self, recommender):
        result = recommender.get_similar_books("B001", top_n=5)
        assert "B001" not in result["book_id"].values

    def test_scores_between_zero_and_one(self, recommender):
        result = recommender.get_similar_books("B001", top_n=5)
        assert (result["similarity_score"] >= 0).all()
        assert (result["similarity_score"] <= 1).all()

    def test_scores_descending_order(self, recommender):
        result = recommender.get_similar_books("B001", top_n=5)
        scores = result["similarity_score"].tolist()
        assert scores == sorted(scores, reverse=True)

    def test_invalid_book_id_raises(self, recommender):
        with pytest.raises(ValueError):
            recommender.get_similar_books("INVALID_ID")

    def test_ai_book_recommends_ai_books(self, recommender):
        """Deep Learning (B001) should recommend other ML/AI books."""
        result = recommender.get_similar_books("B001", top_n=5)
        ai_categories = {"Artificial Intelligence", "Machine Learning", "Computer Vision",
                         "Natural Language Processing"}
        recommended_categories = set(result["category"].values)
        assert len(ai_categories & recommended_categories) > 0


class TestUserRecommendations:
    def test_returns_dataframe(self, recommender):
        result = recommender.recommend_for_user("U001", top_n=5)
        assert isinstance(result, pd.DataFrame)

    def test_borrowed_books_excluded(self, recommender):
        """Recommended books should not include already-borrowed books."""
        history = recommender.get_user_history("U001")
        borrowed_ids = set(history["book_id"].values)
        recs = recommender.recommend_for_user("U001", top_n=5)
        assert not any(bid in borrowed_ids for bid in recs["book_id"].values)

    def test_invalid_user_raises(self, recommender):
        with pytest.raises(ValueError):
            recommender.recommend_for_user("GHOST_USER")

    def test_correct_number_of_results(self, recommender):
        result = recommender.recommend_for_user("U001", top_n=3)
        assert len(result) <= 3


class TestUserHistory:
    def test_history_not_empty(self, recommender):
        history = recommender.get_user_history("U001")
        assert not history.empty

    def test_history_has_title_column(self, recommender):
        history = recommender.get_user_history("U001")
        assert "title" in history.columns
