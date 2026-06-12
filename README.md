# 📚 ML-Based Book Recommendation System

A **content-based filtering** recommendation engine for RFID-integrated library systems. Generates personalized, topic-aware book suggestions from a user's borrowing history using **TF-IDF vectorization** and **cosine similarity**.

Built as part of a smart library automation project — originally replacing a frequency-based approach that recommended only popular books regardless of a reader's interests.

---

##  How It Works?

```
RFID Book Issue
      ↓
Store User Borrowing History
      ↓
Extract Book Metadata (Title + Author + Category + Tags)
      ↓
TF-IDF Vectorization  (sklearn)
      ↓
Pairwise Cosine Similarity Matrix
      ↓
Aggregate Scores Across User History
      ↓
Filter Already-Read Books → Rank → Top-N Recommendations
```

### Why Content-Based over Frequency-Based?

| | Frequency-Based (old) | Content-Based (new) |
|---|---|---|
| Personalization | ❌ Same for everyone | ✅ Per-user |
| Topic awareness | ❌ Popular books dominate | ✅ Matches reading interests |
| Cold start (new books) | ❌ Never recommended | ✅ Works immediately |
| Edge deployment | ✅ | ✅ Lightweight, runs on Raspberry Pi |

---

## Project Structure:

```
book-recommender/
├── data/
│   ├── books.csv               # Book catalogue (ID, title, author, category, tags)
│   └── borrowing_history.csv   # User borrowing records (linked via RFID)
├── src/
│   └── recommender.py          # Core BookRecommender class
├── tests/
│   └── test_recommender.py     # 16 unit tests (pytest)
├── main.py                     # CLI interface
├── requirements.txt
└── README.md
```

---


## 🛠️ Usage in Code

```python
from src.recommender import BookRecommender

rec = BookRecommender(
    books_path="data/books.csv",
    history_path="data/borrowing_history.csv"
)

# Recommend for a user which is based on full borrowing history
recs = rec.recommend_for_user("U001", top_n=5)
print(recs)

# Find books with similar genre
similar = rec.get_similar_books("B001", top_n=5)
print(similar)
```

---

## Example Output

```
👤 User: U001
   Borrowed: Deep Learning, Machine Learning Basics, Python Programming

📚 Top 3 Recommendations
------------------------------------------------------------
  1. [B006] Pattern Recognition and Machine Learning
     Author   : Christopher Bishop
     Category : Machine Learning
     Score    : 0.9135

  2. [B010] Natural Language Processing with Python
     Author   : Steven Bird
     Category : Natural Language Processing
     Score    : 0.3498

  3. [B009] Data Structures and Algorithms in Python
     Author   : Michael Goodrich
     Category : Programming
     Score    : 0.3017
```

---





## 🔧 Tech Stack

- **Python 3.8+**
- **scikit-learn** — TF-IDF vectorization, cosine similarity
- **pandas** — data manipulation
- **numpy** — matrix operations
- **pytest** — unit testing

---
