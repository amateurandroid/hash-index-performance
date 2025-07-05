import pandas as pd
import time
import random
import matplotlib.pyplot as plt
import cProfile
import numpy as np
import json

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
    patterns_names = ["Random", "Sequential", "Clustered", 'Mixed', 'Missing']

    repetitions = 10  # Number of times to repeat each timing for statistics
    # Prepare storage for stats
    stats_hash = {name: {'mean': [], 'min': [], 'max': [], 'std': []} for name in patterns_names}
    stats_linear = {name: {'mean': [], 'min': [], 'max': [], 'std': []} for name in patterns_names}

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
            def record_stats(stats_dict, pattern, durations):
                stats_dict[pattern]['mean'].append(np.mean(durations))
                stats_dict[pattern]['min'].append(np.min(durations))
                stats_dict[pattern]['max'].append(np.max(durations))
                stats_dict[pattern]['std'].append(np.std(durations))

            record_stats(stats_hash, pattern_name, hash_durations)
            record_stats(stats_linear, pattern_name, linear_durations)
            print(f"\nNumber of searches: {n} | Pattern: {pattern_name}")
            print(f"  Hash Index: mean={np.mean(hash_durations):.6f}s, min={np.min(hash_durations):.6f}s, max={np.max(hash_durations):.6f}s, std={np.std(hash_durations):.6f}s")
            print(f"  Linear Search: mean={np.mean(linear_durations):.6f}s, min={np.min(linear_durations):.6f}s, max={np.max(linear_durations):.6f}s, std={np.std(linear_durations):.6f}s")

    def convert_np(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        return obj
    results = {
        'search_sizes': search_sizes,
        'stats_hash': stats_hash,
        'stats_linear': stats_linear
    }
    with open('results.json', 'w') as f:
        json.dump(results, f, default=convert_np, indent=2)
    print('Results saved to results.json')

    def plot_search_performance(stats, search_sizes, patterns_names, title_prefix, marker_style):
        plt.figure(figsize=(12, 10))
        for pattern_name in patterns_names:
            plt.errorbar(
                search_sizes,
                stats[pattern_name]['mean'],
                yerr=stats[pattern_name]['std'],
                marker=marker_style,
                label=f'{title_prefix} ({pattern_name})'
            )
        plt.xlabel('Number of Searches')
        plt.ylabel('Time (seconds)')
        plt.title(f'{title_prefix}: Mean and Std of Search Times by Pattern')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    plot_search_performance(stats_hash, search_sizes, patterns_names, 'Hash Index', 'o')
    plot_search_performance(stats_linear, search_sizes, patterns_names, 'Linear Search', 'x')

    speedup = {}
    for pattern in patterns_names:
        speedup[pattern] = [l/h if h > 0 else None for l, h in zip(stats_linear[pattern]['mean'], stats_hash[pattern]['mean'])]

    plt.figure(figsize=(10, 6))
    for pattern in patterns_names:
        plt.plot(search_sizes, speedup[pattern], marker='o', label=f'Speedup ({pattern})')
    plt.xlabel('Number of Searches')
    plt.ylabel('Linear Search Time / Hash Index Time')
    plt.title('Speedup of Hash Index over Linear Search')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    for stat in ['min', 'max', 'std']:
        plt.figure(figsize=(10, 6))
    for pattern in patterns_names:
        plt.plot(search_sizes, [stats_hash[pattern][stat][i] for i in range(len(search_sizes))], marker='o', label=f'{pattern}')
    plt.xlabel('Number of Searches')
    plt.ylabel(f'{stat.capitalize()} Time (seconds)')
    plt.title(f'Hash Index: {stat.capitalize()} Search Time by Pattern')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

    # Min, Max, Std for Linear Search
    for stat in ['min', 'max', 'std']:
        plt.figure(figsize=(10, 6))
        for pattern in patterns_names:
            plt.plot(search_sizes, [stats_linear[pattern][stat][i] for i in range(len(search_sizes))], marker='x', label=f'{pattern}')
        plt.xlabel('Number of Searches')
        plt.ylabel(f'{stat.capitalize()} Time (seconds)')
        plt.title(f'Linear Search: {stat.capitalize()} Search Time by Pattern')
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    plt.figure(figsize=(10, 6))
    data = [stats_hash[pattern]['mean'] for pattern in patterns_names]
    plt.boxplot(data, labels=patterns_names)
    plt.ylabel('Time (seconds)')
    plt.title('Hash Index: Boxplot of Mean Search Times (across patterns)')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for pattern in patterns_names:
        plt.plot(search_sizes, stats_hash[pattern]['mean'], marker='o', label=f'Hash Index ({pattern})')
        plt.plot(search_sizes, stats_linear[pattern]['mean'], marker='x', label=f'Linear Search ({pattern})')
    plt.xscale('log')
    plt.yscale('log')
    plt.xlabel('Number of Searches (log scale)')
    plt.ylabel('Time (seconds, log scale)')
    plt.title('Log-Log Plot: Search Performance Scaling')
    plt.legend()
    plt.grid(True, which='both')
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    bar_width = 0.35
    index = np.arange(len(patterns_names))
    hash_means = [stats_hash[pattern]['mean'][-1] for pattern in patterns_names]
    linear_means = [stats_linear[pattern]['mean'][-1] for pattern in patterns_names]
    plt.bar(index, hash_means, bar_width, label='Hash Index')
    plt.bar(index + bar_width, linear_means, bar_width, label='Linear Search')
    plt.xlabel('Pattern')
    plt.ylabel('Time (seconds)')
    plt.title('Search Time by Pattern (Largest Search Size)')
    plt.xticks(index + bar_width / 2, patterns_names)
    plt.legend()
    plt.tight_layout()
    plt.show()

    plt.figure(figsize=(10, 6))
    for pattern in patterns_names:
        hash_per_search = [m/n for m, n in zip(stats_hash[pattern]['mean'], search_sizes)]
        linear_per_search = [m/n for m, n in zip(stats_linear[pattern]['mean'], search_sizes)]
        plt.plot(search_sizes, hash_per_search, marker='o', label=f'Hash Index ({pattern})')
        plt.plot(search_sizes, linear_per_search, marker='x', label=f'Linear Search ({pattern})')
    plt.xlabel('Number of Searches')
    plt.ylabel('Time per Search (seconds)')
    plt.title('Average Time per Search by Pattern')
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
