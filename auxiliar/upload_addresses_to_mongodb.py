import pandas as pd
from pymongo import MongoClient
import os
from dotenv import load_dotenv

# Load environment variables (for MONGODB_URI)
load_dotenv()

# Connect to MongoDB
uri = os.getenv("MONGODB_URI")
client = MongoClient(uri)
db = client['complaints_db']
collection = db['addresses']

# Read the CSV
df = pd.read_csv('addresses_clean.csv')

# Convert DataFrame to dictionary records
records = df.to_dict(orient='records')

# Insert into MongoDB (replace existing collection)
collection.delete_many({})  # Optional: clear existing data
collection.insert_many(records)

print(f"Inserted {len(records)} addresses into MongoDB.")

client.close() 