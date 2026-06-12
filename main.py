"""
main.py
-------
Command-line interface for the ML-Based Book Recommendation System.
Simulates RFID book issuance and generates recommendations.

Usage:
    python main.py --user U001 --top 5
    python main.py --book B001 --top 5
    python main.py --demo
"""

import argparse
import sys
from pathlib import Path

# Allow running from project root
sys.path.insert(0, str(Path(__file__).parent))

from src.recommender import BookRecommender

DATA_DIR = Path(__file__).parent / "data"
BOOKS_PATH = DATA_DIR / "books.csv"
HISTORY_PATH = DATA_DIR / "borrowing_history.csv"


def print_banner():
    print("=" * 60)
    print("   ML-Based Book Recommendation System")
    print("   Content-Based Filtering | TF-IDF + Cosine Similarity")
    print("=" * 60)


def display_recommendations(df, label: str):
    print(f"\n📚 {label}")
    print("-" * 60)
    if df.empty:
        print("  No recommendations found.")
        return
    for i, row in df.iterrows():
        score_col = "similarity_score" if "similarity_score" in df.columns else "aggregated_score"
        print(f"  {i+1}. [{row['book_id']}] {row['title']}")
        print(f"     Author   : {row['author']}")
        print(f"     Category : {row['category']}")
        print(f"     Score    : {row[score_col]:.4f}")
        print()


def run_demo(recommender: BookRecommender):
    """Run a demonstration with sample users and books."""
    print("\n[DEMO MODE]")

    # User-based recommendations
    for user_id in ["U001", "U003", "U004"]:
        try:
            history = recommender.get_user_history(user_id)
            borrowed_titles = history["title"].tolist()
            print(f"\n👤 User: {user_id}")
            print(f"   Borrowed: {', '.join(borrowed_titles)}")

            recs = recommender.recommend_for_user(user_id, top_n=3)
            display_recommendations(recs, f"Top 3 Recommendations for {user_id}")
        except ValueError as e:
            print(f"  Error: {e}")

    # Book-based similarity
    print("\n[BOOK SIMILARITY DEMO]")
    for book_id in ["B001", "B013"]:
        try:
            book_name = recommender.books_df[
                recommender.books_df["book_id"] == book_id
            ]["title"].values[0]
            sims = recommender.get_similar_books(book_id, top_n=3)
            display_recommendations(sims, f"Books Similar to '{book_name}'")
        except ValueError as e:
            print(f"  Error: {e}")


def main():
    parser = argparse.ArgumentParser(
        description="ML-Based Book Recommendation System"
    )
    parser.add_argument("--user", type=str, help="User ID to generate recommendations for (e.g. U001)")
    parser.add_argument("--book", type=str, help="Book ID to find similar books for (e.g. B001)")
    parser.add_argument("--top", type=int, default=5, help="Number of recommendations (default: 5)")
    parser.add_argument("--demo", action="store_true", help="Run demo with sample data")
    args = parser.parse_args()

    print_banner()

    recommender = BookRecommender(
        books_path=str(BOOKS_PATH),
        history_path=str(HISTORY_PATH),
    )
    print(f"\n✅ Loaded catalogue: {recommender.catalogue_size()} books")

    if args.demo:
        run_demo(recommender)

    elif args.user:
        try:
            history = recommender.get_user_history(args.user)
            borrowed_titles = history["title"].tolist()
            print(f"\n👤 User: {args.user}")
            print(f"   Borrowing History: {', '.join(borrowed_titles)}")
            recs = recommender.recommend_for_user(args.user, top_n=args.top)
            display_recommendations(recs, f"Top {args.top} Recommendations")
        except ValueError as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)

    elif args.book:
        try:
            book_name = recommender.books_df[
                recommender.books_df["book_id"] == args.book
            ]["title"].values[0]
            sims = recommender.get_similar_books(args.book, top_n=args.top)
            display_recommendations(sims, f"Books Similar to '{book_name}'")
        except ValueError as e:
            print(f"\n❌ Error: {e}")
            sys.exit(1)

    else:
        print("\nNo arguments provided. Running demo...\n")
        run_demo(recommender)


if __name__ == "__main__":
    main()
