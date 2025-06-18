# Hash Index vs Linear Search - Performance Test

## 📁 Dataset

[Product Classification and Clustering](https://archive.ics.uci.edu/dataset/837/product+classification+and+clustering)

- 35,000+ product entries from a comparison platform
- Used `Product ID` for indexing

## ⚙️ What We Did

- Implemented a **Hash Index** using Python dictionary
- Searched for 100 random product IDs
- Compared performance against traditional **Linear Search** using pandas filtering

## ⏱️ Results

| Method        | Time Taken (100 lookups) |
| ------------- | ------------------------ |
| Hash Index    | 0.000047 seconds         |
| Linear Search | 0.011121 seconds         |

**Hash Index is ~236x faster than linear search**

## 🧠 Why Hash Index

- O(1) average time complexity
- Perfect for exact-match lookups like `Product ID`

## ▶️ How to Run

- Requires Python 3
- Install dependencies: `pip install -r requirements.txt`
- Run with memory profiling: `python -m memory_profiler main.py`
- Run normally: `python main.py`
