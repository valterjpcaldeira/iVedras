#!/usr/bin/env python3
"""
Address Data Processing Script for iVedras
This script processes the moradas_torres_vedras.csv file to clean and prepare it for Snowflake.
"""

import pandas as pd
import re
import numpy as np
from datetime import datetime

def clean_latitude(lat_str):
    """Clean latitude value by removing GPS: prefix and converting to float"""
    if pd.isna(lat_str):
        return None
    
    lat_str = str(lat_str).strip()
    if lat_str.startswith('GPS:'):
        lat_str = lat_str.replace('GPS:', '').strip()
    
    try:
        lat = float(lat_str)
        return lat
    except (ValueError, TypeError):
        return None

def clean_longitude(lon_str):
    """Clean longitude value and convert to float"""
    if pd.isna(lon_str):
        return None
    
    try:
        lon = float(str(lon_str).strip())
        return lon
    except (ValueError, TypeError):
        return None

def normalize_address(address):
    """Normalize address for comparison (lowercase, remove punctuation)"""
    if pd.isna(address):
        return ""
    
    # Convert to lowercase and remove punctuation
    normalized = re.sub(r'[^\w\s]', '', str(address).lower())
    # Remove extra whitespace
    normalized = ' '.join(normalized.split())
    return normalized

def clean_address(address):
    """Clean address while preserving original format"""
    if pd.isna(address):
        return ""
    
    # Remove newlines and extra whitespace
    cleaned = re.sub(r'\n+', ' ', str(address))
    cleaned = ' '.join(cleaned.split())
    return cleaned

def is_torres_vedras_coordinate(lat, lon):
    """Check if coordinates are within Torres Vedras area"""
    if lat is None or lon is None:
        return False
    
    # Torres Vedras approximate boundaries
    # Latitude: 38.8 to 39.3 (roughly)
    # Longitude: -9.5 to -9.0 (roughly)
    return (38.8 <= lat <= 39.3) and (-9.5 <= lon <= -9.0)

def process_addresses_file():
    """Process the addresses file and create cleaned version"""
    print("🚀 Starting address data processing...")
    
    # Read the original file
    print("📖 Reading moradas_torres_vedras.csv...")
    try:
        df = pd.read_csv('moradas_torres_vedras.csv')
        print(f"📊 Original data shape: {df.shape}")
        print(f"📋 Original columns: {list(df.columns)}")
        
        # Show first few rows
        print("\n📋 First few rows of original data:")
        print(df.head())
        
    except FileNotFoundError:
        print("❌ File moradas_torres_vedras.csv not found!")
        return
    except Exception as e:
        print(f"❌ Error reading file: {e}")
        return
    
    # Identify latitude and longitude columns
    lat_col = None
    lon_col = None
    
    for col in df.columns:
        if 'lat' in col.lower():
            lat_col = col
        elif 'lon' in col.lower():
            lon_col = col
    
    if lat_col and lon_col:
        print(f"📍 Found latitude column: {lat_col}")
        print(f"📍 Found longitude column: {lon_col}")
    else:
        print("❌ Could not identify latitude/longitude columns!")
        return
    
    # Clean the data
    print("\n🧹 Cleaning data...")
    
    # Remove rows with missing addresses
    df = df.dropna(subset=['address'])
    
    # Clean coordinates
    df['latitude'] = df[lat_col].apply(clean_latitude)
    df['longitude'] = df[lon_col].apply(clean_longitude)
    
    # Remove rows with invalid coordinates
    df = df.dropna(subset=['latitude', 'longitude'])
    
    # Filter for Torres Vedras coordinates only
    df['is_torres_vedras'] = df.apply(
        lambda row: is_torres_vedras_coordinate(row['latitude'], row['longitude']), 
        axis=1
    )
    
    # Keep only Torres Vedras coordinates
    df_filtered = df[df['is_torres_vedras']].copy()
    
    # Clean addresses
    df_filtered['address'] = df_filtered['address'].apply(clean_address)
    df_filtered['normalized_address'] = df_filtered['address'].apply(normalize_address)
    
    # Add ID and timestamp
    df_filtered['address_id'] = range(1, len(df_filtered) + 1)
    df_filtered['processed_at'] = datetime.now()
    
    # Select and reorder columns
    final_columns = ['address_id', 'address', 'latitude', 'longitude', 'normalized_address', 'processed_at']
    df_final = df_filtered[final_columns].copy()
    
    # Remove duplicates based on normalized address
    df_final = df_final.drop_duplicates(subset=['normalized_address'])
    
    # Reset address_id
    df_final['address_id'] = range(1, len(df_final) + 1)
    
    print(f"\n🧹 Data cleaning results:")
    print(f"   Initial rows: {len(df)}")
    print(f"   After coordinate filtering: {len(df_filtered)}")
    print(f"   After duplicate removal: {len(df_final)}")
    print(f"   Removed rows: {len(df) - len(df_final)}")
    
    # Show sample of cleaned data
    print(f"\n📋 Sample of cleaned data:")
    print(df_final.head())
    
    # Save cleaned data
    print(f"\n💾 Saving cleaned data to addresses_clean.csv...")
    df_final.to_csv('addresses_clean.csv', index=False)
    
    # Data statistics
    print(f"\n📈 Data statistics:")
    print(f"   Total addresses: {len(df_final)}")
    print(f"   Unique addresses: {df_final['normalized_address'].nunique()}")
    print(f"   Latitude range: {df_final['latitude'].min():.6f} to {df_final['latitude'].max():.6f}")
    print(f"   Longitude range: {df_final['longitude'].min():.6f} to {df_final['longitude'].max():.6f}")
    
    # Data quality checks
    print(f"\n🔍 Data quality checks:")
    duplicates = df_final[df_final.duplicated(subset=['normalized_address'], keep=False)]
    print(f"   ⚠️ Found {len(duplicates)} duplicate addresses")
    
    outside_range = df_final[~df_final.apply(
        lambda row: is_torres_vedras_coordinate(row['latitude'], row['longitude']), 
        axis=1
    )]
    print(f"   ⚠️ Found {len(outside_range)} addresses with coordinates outside Torres Vedras range")
    
    print(f"\n✅ Processing complete! Cleaned data saved to addresses_clean.csv")
    
    # Create Snowflake insert script
    print(f"\n📝 Creating Snowflake insert script for table 'addresses'...")
    create_snowflake_insert_script(df_final)
    
    print(f"\n🎉 All processing complete!")
    print(f"📁 Files created:")
    print(f"   - addresses_clean.csv (cleaned data)")
    print(f"   - snowflake_insert_addresses.sql (SQL insert script)")

def create_snowflake_insert_script(df):
    """Create SQL insert script for Snowflake"""
    sql_lines = [
        "-- Snowflake Insert Script for Addresses",
        "-- Generated on: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "",
        "CREATE OR REPLACE TABLE addresses (",
        "    address_id INTEGER,",
        "    address STRING,",
        "    latitude FLOAT,",
        "    longitude FLOAT,",
        "    normalized_address STRING,",
        "    processed_at TIMESTAMP_NTZ",
        ");",
        "",
        "INSERT INTO addresses (address_id, address, latitude, longitude, normalized_address, processed_at) VALUES"
    ]
    
    # Add data rows
    for idx, row in df.iterrows():
        # Escape single quotes in address
        address = row['address'].replace("'", "''")
        normalized = row['normalized_address'].replace("'", "''")
        
        sql_line = f"({row['address_id']}, '{address}', {row['latitude']}, {row['longitude']}, '{normalized}', '{row['processed_at']}')"
        
        if idx < len(df) - 1:
            sql_line += ","
        else:
            sql_line += ";"
        
        sql_lines.append(sql_line)
    
    # Write to file
    with open('snowflake_insert_addresses.sql', 'w', encoding='utf-8') as f:
        f.write('\n'.join(sql_lines))
    
    print("✅ SQL script saved to snowflake_insert_addresses.sql")

if __name__ == "__main__":
    process_addresses_file() 