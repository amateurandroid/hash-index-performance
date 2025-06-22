import pandas as pd
import time
import random
import matplotlib.pyplot as plt
import cProfile
import numpy as np

def build_hash_index(df):
    hash_index = {}
    for idx, row in df.iterrows():
        product_id = row['Product ID']
        hash_index[product_id] = row
    return hash_index

def hash_index_search(hash_index, product_ids):
    for pid in product_ids:
        _ = hash_index.get(pid, None)

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
    # Prepare timing storage for each pattern
    patterns_names = ["Random", "Sequential", "Clustered"]

    repetitions = 7  # Number of times to repeat each timing for statistics
    # Prepare storage for stats
    stats_hash = {name: {'mean': [], 'min': [], 'max': [], 'std': []} for name in patterns_names + ['Mixed', 'Missing']}
    stats_linear = {name: {'mean': [], 'min': [], 'max': [], 'std': []} for name in patterns_names + ['Mixed', 'Missing']}

    print("\nPerformance comparison for different numbers of searches (sequential, random, clustered, mixed, missing):")

    for n in search_sizes:
        n = min(n, len(df["Product ID"]))
        product_ids = df["Product ID"].tolist()
        max_id = max(product_ids)
        min_id = min(product_ids)

        # Query patterns
        random_ids = random.sample(product_ids, n)
        # Sequential queries: take a contiguous range
        start_idx = random.randint(0, len(product_ids) - n)
        sequential_ids = product_ids[start_idx:start_idx + n]
        # Clustered queries: pick k clusters of size n//k
        k = min(5, n)  # number of clusters
        cluster_size = n // k if k > 0 else n
        clustered_ids = []
        for _ in range(k):
            cluster_start = random.randint(0, len(product_ids) - cluster_size)
            clustered_ids.extend(product_ids[cluster_start:cluster_start + cluster_size])
        clustered_ids = clustered_ids[:n]  # ensure length n
        # Mixed queries: half existing, half missing
        existing_ids = random_ids[:n//2]
        missing_ids = [max_id + i + 1 for i in range(n - n//2)]
        mixed_ids = existing_ids + missing_ids
        # All missing queries
        all_missing_ids = [max_id + i + 1 for i in range(n)]

        pattern_dict = {
            'Random': random_ids,
            'Sequential': sequential_ids,
            'Clustered': clustered_ids,
            'Mixed': mixed_ids,
            'Missing': all_missing_ids
        }

        for pattern_name, ids in pattern_dict.items():
            hash_durations = []
            linear_durations = []
            for _ in range(repetitions):
                start_time = time.time()
                hash_index_search(hash_index, ids)
                hash_durations.append(time.time() - start_time)
                start_time = time.time()
                linear_search(df, ids)
                linear_durations.append(time.time() - start_time)
            # Store stats
            stats_hash[pattern_name]['mean'].append(np.mean(hash_durations))
            stats_hash[pattern_name]['min'].append(np.min(hash_durations))
            stats_hash[pattern_name]['max'].append(np.max(hash_durations))
            stats_hash[pattern_name]['std'].append(np.std(hash_durations))
            stats_linear[pattern_name]['mean'].append(np.mean(linear_durations))
            stats_linear[pattern_name]['min'].append(np.min(linear_durations))
            stats_linear[pattern_name]['max'].append(np.max(linear_durations))
            stats_linear[pattern_name]['std'].append(np.std(linear_durations))
            print(f"\nNumber of searches: {n} | Pattern: {pattern_name}")
            print(f"  Hash Index: mean={np.mean(hash_durations):.6f}s, min={np.min(hash_durations):.6f}s, max={np.max(hash_durations):.6f}s, std={np.std(hash_durations):.6f}s")
            print(f"  Linear Search: mean={np.mean(linear_durations):.6f}s, min={np.min(linear_durations):.6f}s, max={np.max(linear_durations):.6f}s, std={np.std(linear_durations):.6f}s")

    plt.figure(figsize=(12, 7))
    for pattern_name in patterns_names + ['Mixed', 'Missing']:
        plt.errorbar(search_sizes, stats_hash[pattern_name]['mean'], yerr=stats_hash[pattern_name]['std'], marker='o', label=f'Hash Index ({pattern_name})')
    plt.xlabel('Number of Searches')
    plt.ylabel('Time (seconds)')
    plt.title('Hash Index Search: Mean and Std of Search Times by Pattern')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(12, 7))
    for pattern_name in patterns_names + ['Mixed', 'Missing']:
        plt.errorbar(search_sizes, stats_linear[pattern_name]['mean'], yerr=stats_linear[pattern_name]['std'], marker='x', label=f'Linear Search ({pattern_name})')
    plt.xlabel('Number of Searches')
    plt.ylabel('Time (seconds)')
    plt.title('Linear Search: Mean and Std of Search Times by Pattern')
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
    stats.strip_dirs().sort_stats('cumulative').print_stats('main.py', 40)
    stats.print_callees('main.py')
