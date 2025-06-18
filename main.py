import pandas as pd
import time
import random
import matplotlib.pyplot as plt
from memory_profiler import profile
import cProfile

#Loop over the DataFrame and populate the index
@profile
def build_hash_index(df):
    hash_index = {}
    for idx, row in df.iterrows():
        product_id = row['Product ID']
        hash_index[product_id] = row
    return hash_index

@profile
def hash_index_search(hash_index, product_ids):
    for pid in product_ids:
        _ = hash_index.get(pid, None)

@profile
def linear_search(df, product_ids):
    for pid in product_ids:
        _ = df[df["Product ID"] == pid]

def main():
    #Reading the CSV
    df = pd.read_csv("pricerunner_aggregate.csv")

    #Striping extra spaces in column names
    df.columns = df.columns.str.strip()

    #Printing first 5 rows
    print(df.head())

    #Printing column names
    print("\nColumns in the dataset:")
    print(df.columns)

    #Simple hash index using a dictionary
    hash_index = {}

    hash_index = build_hash_index(df)

    #Searching by product ID
    search_id = 3
    result = hash_index.get(search_id, "Not found")

    print(f"\nSearch result for Product ID {search_id}:")
    print(result)

    #Performance Comparison

    # Timing the hash index search
    search_sizes = [10, 100, 1000, 5000, 10000]
    hash_times_existing = []
    linear_times_existing = []
    hash_times_mixed = []
    linear_times_mixed = []

    print("\nPerformance comparison for different numbers of searches:")

    for n in search_sizes:
        n = min(n, len(df["Product ID"]))
        existing_ids = random.sample(df["Product ID"].tolist(), n)
        max_id = max(df["Product ID"])
        missing_ids = [max_id + i + 1 for i in range(n)]
        mixed_ids = existing_ids[:n//2] + missing_ids[:n - n//2]

        # Hash index search timing (existing)
        start_time = time.time()
        hash_index_search(hash_index, existing_ids)
        hash_duration_existing = time.time() - start_time
        hash_times_existing.append(hash_duration_existing)

        # Linear search timing (existing)
        start_time = time.time()
        linear_search(df, existing_ids)
        linear_duration_existing = time.time() - start_time
        linear_times_existing.append(linear_duration_existing)

        # Hash index search timing (mixed)
        start_time = time.time()
        hash_index_search(hash_index, mixed_ids)
        hash_duration_mixed = time.time() - start_time
        hash_times_mixed.append(hash_duration_mixed)

        # Linear search timing (mixed)
        start_time = time.time()
        linear_search(df, mixed_ids)
        linear_duration_mixed = time.time() - start_time
        linear_times_mixed.append(linear_duration_mixed)

        print(f"\nNumber of searches: {n}")
        print(f"  Hash Index (existing): {hash_duration_existing:.6f} seconds")
        print(f"  Linear Search (existing): {linear_duration_existing:.6f} seconds")
        print(f"  Hash Index (mixed): {hash_duration_mixed:.6f} seconds")
        print(f"  Linear Search (mixed): {linear_duration_mixed:.6f} seconds")

    # Plotting the results for existing IDs
    plt.figure(figsize=(8, 5))
    plt.plot(search_sizes, hash_times_existing, marker='o', label='Hash Index Search (existing)')
    plt.plot(search_sizes, linear_times_existing, marker='o', label='Linear Search (existing)')
    plt.xlabel('Number of Searches')
    plt.ylabel('Time (seconds)')
    plt.title('Search Performance (All IDs Exist)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Plotting the results for mixed IDs
    plt.figure(figsize=(8, 5))
    plt.plot(search_sizes, hash_times_mixed, marker='o', label='Hash Index Search (mixed)')
    plt.plot(search_sizes, linear_times_mixed, marker='o', label='Linear Search (mixed)')
    plt.xlabel('Number of Searches')
    plt.ylabel('Time (seconds)')
    plt.title('Search Performance (Half IDs Missing)')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    import cProfile
    import pstats

    with cProfile.Profile() as pr:
        main()
    stats = pstats.Stats(pr)
    # Sort by cumulative time and print top 20 lines
    stats.strip_dirs().sort_stats('cumulative').print_stats(20)