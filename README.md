# 📚 ML-Based Book Recommendation System

A **content-based filtering** recommendation engine for RFID-integrated library systems. Generates personalized, topic-aware book suggestions from a user's borrowing history using **TF-IDF vectorization** and **cosine similarity**.

Built as part of a smart library automation project — originally replacing a frequency-based approach that recommended only popular books regardless of a reader's interests.

---

## 🧠 How It Works

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

## 📁 Project Structure

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

## 🚀 Getting Started

### 1. Clone & Install

```bash
git clone https://github.com/<your-username>/book-recommender.git
cd book-recommender
pip install -r requirements.txt
```

### 2. Run the Demo

```bash
python main.py --demo
```

### 3. Get Recommendations for a User

```bash
python main.py --user U001 --top 5
```

### 4. Find Books Similar to a Specific Book

```bash
python main.py --book B001 --top 5
```

---

## 🛠️ Usage in Code

```python
from src.recommender import BookRecommender

rec = BookRecommender(
    books_path="data/books.csv",
    history_path="data/borrowing_history.csv"
)

# Recommend for a user (based on full borrowing history)
recs = rec.recommend_for_user("U001", top_n=5)
print(recs)

# Find books similar to a specific book
similar = rec.get_similar_books("B001", top_n=5)
print(similar)
```

---

## 📊 Example Output

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

## 🧪 Tests

```bash
pytest tests/ -v
```

16 tests covering: data loading, similarity scores, ranking order, exclusion of already-read books, error handling, and semantic correctness (ML books recommend ML books).

---

## 📦 Data Format

**books.csv**
```csv
book_id,title,author,category,tags
B001,Deep Learning,Ian Goodfellow,Artificial Intelligence,deep learning neural networks AI
```

**borrowing_history.csv**
```csv
user_id,book_id,issue_date,return_date
U001,B001,2024-01-05,2024-01-20
```

Replace with your own library data. The `user_id` maps to an RFID card/tag ID.

---

## 🔧 Tech Stack

- **Python 3.8+**
- **scikit-learn** — TF-IDF vectorization, cosine similarity
- **pandas** — data manipulation
- **numpy** — matrix operations
- **pytest** — unit testing

---

## 📄 Resume Description

> Developed a content-based book recommendation system using TF-IDF feature extraction and cosine similarity to deliver personalized, topic-aware recommendations from RFID-tracked borrowing history; replaced a frequency-based approach, improving recommendation relevance and enabling lightweight edge deployment on Raspberry Pi.
