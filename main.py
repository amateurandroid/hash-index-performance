import pandas as pd
import time
import random

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

#Loop over the DataFrame and populate the index
for idx, row in df.iterrows():
    product_id = row['Product ID']
    hash_index[product_id] = row

#Searching by product ID
search_id = 3
result = hash_index.get(search_id, "Not found")

print(f"\nSearch result for Product ID {search_id}:")
print(result)

#Performance Comparison
#100 random product IDs
random_ids = random.sample(df["Product ID"].tolist(), 100)

#Hash index search
start_time = time.time()

for pid in random_ids:
    _ = hash_index.get(pid, None)

hash_duration = time.time() - start_time
print(f"\nTime to search 100 product IDs using Hash Index: {hash_duration:.6f} seconds")

#Linear Search
start_time = time.time()

for pid in random_ids:
    _ = df[df["Product ID"] == pid]

linear_duration = time.time() - start_time
print(f"Time to search 100 product IDs using Linear Search: {linear_duration:.6f} seconds")
