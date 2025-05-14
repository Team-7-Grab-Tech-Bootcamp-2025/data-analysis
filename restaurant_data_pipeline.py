import pandas as pd
import numpy as np
import re
import os
import time
from datetime import datetime

# Define all paths used throughout the pipeline
DATA_PATHS = {
    # Input data directories
    'raw_befood': './raw_data/befood',
    'raw_foody': './raw_data/foody',
    'cleaned_data': './cleaned_data',
    
    # Output directories for each stage
    'initial_output': './output',
    'fixed_references': './output',
    'final_output': './output',
    'validated_data': './output',
    
    # Sentiment data paths
    'befood_sentiment': './sentiment_category/befood_predicted_sentiment.csv',
    'foody_sentiment': './sentiment_category/foody_predicted_sentiment.csv',
}

# Create necessary directories
for directory in set(DATA_PATHS.values()):
    if not os.path.exists(directory):
        os.makedirs(directory)
        print(f"Created directory: {directory}")

print("=== RESTAURANT DATA PROCESSING PIPELINE ===")
print("This script performs the complete pipeline for processing restaurant review data:")
print("1. Initial data processing and table creation")
print("2. Fix reference issues between tables")
print("3. Fix restaurant reference mapping")
print("4. Fix dish IDs format")
print("5. Clean dish data by removing invalid references")
print("6. Validate data integrity with null checks and reference validation")
print("7. Process sentiment labels for feedback")

# ==================== HELPER FUNCTIONS ====================

def standardize_address(address):
    """Standardize restaurant addresses for better matching"""
    if pd.isna(address):
        return address
    
    # Convert to string and lowercase
    address = str(address).lower().strip()
    
    # Remove duplicate spaces
    address = re.sub(r'\s+', ' ', address)
    
    # Standardize common abbreviations
    address = re.sub(r'\bq\.', 'quận', address)
    address = re.sub(r'\bp\.', 'phường', address)
    address = re.sub(r'\bd\.', 'đường', address)
    
    # Standardize common street names
    address = re.sub(r'\bcmtt\b', 'cách mạng tháng tám', address)
    address = re.sub(r'\blvt\b', 'lê văn thiêm', address)
    address = re.sub(r'\blvd\b', 'lê văn duyệt', address)
    
    return address

# ==================== STAGE 1: INITIAL DATA PROCESSING ====================

def process_initial_data():
    """
    Perform initial data processing including:
    - Loading cleaned data
    - Standardizing city names
    - Creating reference tables (City, District, Platform, etc.)
    - Handling restaurant ID mapping and deduplication
    - Creating all output tables according to ER Diagram
    """
    print("\n=== STAGE 1: INITIAL DATA PROCESSING ===")
    output_dir = DATA_PATHS['initial_output']
    
    # Load befood restaurants data
    print("Loading befood restaurants data...")
    befood_restaurants = pd.read_csv(f"{DATA_PATHS['cleaned_data']}/befood_restaurants_cleaned.csv", encoding='utf-8')

    # Load foody restaurants data
    print("Loading foody restaurants data...")
    foody_restaurants = pd.read_csv(f"{DATA_PATHS['cleaned_data']}/foody_restaurants_cleaned.csv", encoding='utf-8')

    # Load raw befood data for food_type
    print("Loading raw befood restaurants data for food_type...")
    try:
        raw_befood_restaurants = pd.read_csv(f"{DATA_PATHS['raw_befood']}/restaurants.csv", encoding='utf-8')
    except UnicodeDecodeError:
        # Try with different encodings if utf-8 fails
        encodings = ['latin1', 'cp1252', 'iso-8859-1']
        for encoding in encodings:
            try:
                raw_befood_restaurants = pd.read_csv(f"{DATA_PATHS['raw_befood']}/restaurants.csv", encoding=encoding)
                print(f"Successfully loaded with {encoding} encoding")
                break
            except UnicodeDecodeError:
                continue

    # Load other data files
    print("Loading dishes data...")
    befood_dishes = pd.read_csv(f"{DATA_PATHS['cleaned_data']}/befood_dishes_cleaned.csv", encoding='utf-8', low_memory=False)
    foody_dishes = pd.read_csv(f"{DATA_PATHS['cleaned_data']}/foody_dishes_cleaned.csv", encoding='utf-8')

    print("Loading reviews data...")
    befood_reviews = pd.read_csv(f"{DATA_PATHS['cleaned_data']}/befood_reviews_cleaned.csv", encoding='utf-8')
    foody_reviews = pd.read_csv(f"{DATA_PATHS['cleaned_data']}/foody_reviews_cleaned.csv", encoding='utf-8')

    # 1. Change city_name from 'TP. Hồ Chí Minh' to 'TP. HCM' in befood
    print("Standardizing city names...")
    befood_restaurants['city_name'] = befood_restaurants['city_name'].replace('TP. Hồ Chí Minh', 'TP. HCM')

    # 2. Remove entries where city_name is not 'TP. HCM' or 'Hà Nội'
    print("Filtering by city...")
    befood_restaurants = befood_restaurants[befood_restaurants['city_name'].isin(['TP. HCM', 'Hà Nội'])]
    foody_restaurants = foody_restaurants[foody_restaurants['city_name'].isin(['TP. HCM', 'Hà Nội'])]

    # Create city table
    print("Creating city table...")
    city_table = pd.DataFrame({
        'city_id': [1, 2],
        'city_name': ['TP. HCM', 'Hà Nội']
    })

    # 3. Create food_type table from merchant_category_name
    print("Creating food_type table...")
    food_type = pd.DataFrame(raw_befood_restaurants[['merchant_category_id', 'merchant_category_name']].drop_duplicates())
    food_type = food_type.rename(columns={'merchant_category_id': 'food_type_id', 'merchant_category_name': 'food_type_name'})

    # Fix the NaN values before converting to int
    food_type['food_type_id'] = food_type['food_type_id'].fillna(0)
    food_type['food_type_id'] = food_type['food_type_id'].astype(int)

    # Add food_type_id to befood_restaurants
    befood_restaurants = pd.merge(
        befood_restaurants,
        raw_befood_restaurants[['restaurant_id', 'merchant_category_id']],
        left_on='restaurant_id',
        right_on='restaurant_id',
        how='left'
    )
    befood_restaurants = befood_restaurants.rename(columns={'merchant_category_id': 'food_type_id'})
    
    # Reassign district_id by sorting district_name in descending order
    print("Reassigning district_id based on sorted district_name...")

    # Get unique district names from both sources
    all_districts = pd.concat([
        befood_restaurants[['district_name']],
        foody_restaurants[['district_name']]
    ]).drop_duplicates()

    # Sort district names in descending order
    all_districts = all_districts.sort_values('district_name', ascending=False)

    # Create new district_id starting from 1
    all_districts['new_district_id'] = range(1, len(all_districts) + 1)

    # Create mapping from old district_name to new district_id
    district_mapping = dict(zip(all_districts['district_name'], all_districts['new_district_id']))
    print(f"Created mapping for {len(district_mapping)} districts")

    # Update district_id in befood_restaurants
    befood_restaurants['district_id'] = befood_restaurants['district_name'].map(district_mapping)

    # Update district_id in foody_restaurants
    foody_restaurants['district_id'] = foody_restaurants['district_name'].map(district_mapping)

    # 4. Standardize addresses and restaurant names for duplicate checking
    print("Standardizing addresses and restaurant names...")
    befood_restaurants['std_address'] = befood_restaurants['address'].apply(standardize_address)
    foody_restaurants['std_address'] = foody_restaurants['address'].apply(standardize_address)
    befood_restaurants['std_name'] = befood_restaurants['restaurant_name'].str.lower().str.strip()
    foody_restaurants['std_name'] = foody_restaurants['restaurant_name'].str.lower().str.strip()

    # 5. Handle restaurant_id and check for duplicates
    print("Handling restaurant IDs and checking for duplicates...")

    # Store original IDs before adding prefixes
    befood_restaurants['original_id'] = befood_restaurants['restaurant_id'].copy()
    foody_restaurants['original_id'] = foody_restaurants['restaurant_id'].copy()

    # Add platform prefix to restaurant_id - convert befood IDs to integers first
    befood_restaurants['restaurant_id'] = 'B_' + befood_restaurants['restaurant_id'].astype(int).astype(str)
    foody_restaurants['restaurant_id'] = 'F_' + foody_restaurants['restaurant_id'].astype(str)

    # Find potential duplicates based on standardized address or name
    befood_addresses = set(befood_restaurants['std_address'].dropna())
    foody_addresses = set(foody_restaurants['std_address'].dropna())
    common_addresses = befood_addresses.intersection(foody_addresses)

    befood_names = set(befood_restaurants['std_name'].dropna())
    foody_names = set(foody_restaurants['std_name'].dropna())
    common_names = befood_names.intersection(foody_names)

    print(f"Found {len(common_addresses)} common addresses and {len(common_names)} common names.")

    # Create mapping for duplicate restaurants
    duplicate_mapping = {}

    # Check for address matches first
    for addr in common_addresses:
        befood_matches = befood_restaurants[befood_restaurants['std_address'] == addr]['restaurant_id'].tolist()
        foody_matches = foody_restaurants[foody_restaurants['std_address'] == addr]['restaurant_id'].tolist()
        
        for f_id in foody_matches:
            duplicate_mapping[f_id] = befood_matches[0] if befood_matches else None

    # Then check for name matches if no address match was found
    for name in common_names:
        befood_matches = befood_restaurants[befood_restaurants['std_name'] == name]['restaurant_id'].tolist()
        foody_matches = foody_restaurants[foody_restaurants['std_name'] == name]['restaurant_id'].tolist()
        
        for f_id in foody_matches:
            if f_id not in duplicate_mapping or duplicate_mapping[f_id] is None:
                duplicate_mapping[f_id] = befood_matches[0] if befood_matches else None

    # Count valid mappings
    valid_mappings = sum(1 for b_id in duplicate_mapping.values() if b_id is not None)
    print(f"Found {valid_mappings} duplicate restaurants between platforms.")

    # Create a copy of reviews and dishes data for processing
    print("Processing reviews and dishes data...")
    befood_reviews_copy = befood_reviews.copy()
    foody_reviews_copy = foody_reviews.copy()
    befood_dishes_copy = befood_dishes.copy()
    foody_dishes_copy = foody_dishes.copy()

    # Add platform prefixes to IDs in reviews and dishes
    befood_reviews_copy['restaurant_id'] = 'B_' + befood_reviews_copy['restaurant_id'].astype(int).astype(str)
    foody_reviews_copy['restaurant_id'] = 'F_' + foody_reviews_copy['restaurant_id'].astype(str)

    # Check if the restaurant_id column in befood_dishes_copy contains only numeric values
    print("Checking befood dishes data for invalid restaurant IDs...")
    befood_dishes_copy['is_numeric'] = pd.to_numeric(befood_dishes_copy['restaurant_id'], errors='coerce').notnull()
    numeric_dishes = befood_dishes_copy[befood_dishes_copy['is_numeric']]
    non_numeric_dishes = befood_dishes_copy[~befood_dishes_copy['is_numeric']]

    if not non_numeric_dishes.empty:
        print(f"Found {len(non_numeric_dishes)} records with non-numeric restaurant IDs in befood dishes.")
        print("Sample of problematic records:")
        print(non_numeric_dishes.head())
        # Keep only numeric restaurant IDs
        befood_dishes_copy = numeric_dishes

    # Now safely convert to int and add prefix
    befood_dishes_copy['restaurant_id'] = 'B_' + befood_dishes_copy['restaurant_id'].astype(float).astype(int).astype(str)
    befood_dishes_copy = befood_dishes_copy.drop('is_numeric', axis=1)
    foody_dishes_copy['restaurant_id'] = 'F_' + foody_dishes_copy['restaurant_id'].astype(str)

    # Add prefixes to rating_id and dish_id
    befood_reviews_copy['rating_id'] = 'BR_' + befood_reviews_copy['rating_id'].astype(str)
    foody_reviews_copy['rating_id'] = 'FR_' + foody_reviews_copy['rating_id'].astype(str)
    befood_dishes_copy['dish_id'] = 'B_' + befood_dishes_copy['dish_id'].astype(str)
    foody_dishes_copy['dish_id'] = 'F_' + foody_dishes_copy['dish_id'].astype(str)

    # Rename review_text to feedback to match the ER diagram
    befood_reviews_copy = befood_reviews_copy.rename(columns={'review_text': 'feedback'})
    foody_reviews_copy = foody_reviews_copy.rename(columns={'review_text': 'feedback'})

    # Create data structure to store ratings from both platforms for duplicate restaurants
    print("Handling duplicate restaurants...")
    duplicate_ratings = []

    # For each duplicate restaurant, store ratings from both platforms and update IDs
    for f_id, b_id in duplicate_mapping.items():
        if b_id:
            # Store Foody platform rating
            foody_restaurant = foody_restaurants[foody_restaurants['restaurant_id'] == f_id]
            if not foody_restaurant.empty:
                duplicate_ratings.append({
                    'restaurant_id': b_id,  # Use Befood ID as common identifier
                    'platform_id': 2,  # Foody platform
                    'restaurant_rating': foody_restaurant.iloc[0]['restaurant_rating']
                })
            
            # Store Befood platform rating
            befood_restaurant = befood_restaurants[befood_restaurants['restaurant_id'] == b_id]
            if not befood_restaurant.empty:
                duplicate_ratings.append({
                    'restaurant_id': b_id,  # Befood ID
                    'platform_id': 1,  # Befood platform
                    'restaurant_rating': befood_restaurant.iloc[0]['restaurant_rating']
                })
            
            # Update restaurant_id in foody reviews to point to Befood equivalent
            foody_reviews_copy.loc[foody_reviews_copy['restaurant_id'] == f_id, 'restaurant_id'] = b_id
            
            # Update restaurant_id in foody dishes to point to Befood equivalent
            foody_dishes_copy.loc[foody_dishes_copy['restaurant_id'] == f_id, 'restaurant_id'] = b_id

    # Create district table from existing data
    print("Creating district table...")
    district_table = pd.concat([
        befood_restaurants[['district_id', 'district_name']],
        foody_restaurants[['district_id', 'district_name']]
    ]).drop_duplicates()

    # Create platform table
    print("Creating platform table...")
    platform_table = pd.DataFrame({
        'platform_id': [1, 2],
        'platform_name': ['befood', 'foody']
    })

    # Create Restaurant table - keep both befood and foody entries
    print("Creating Restaurant table...")
    # For restaurant table, we need to handle duplicates specially
    # We'll include all restaurants but mark duplicates for reference
    foody_restaurants_with_duplicate_flag = foody_restaurants.copy()
    foody_restaurants_with_duplicate_flag['is_duplicate'] = foody_restaurants_with_duplicate_flag['restaurant_id'].isin(duplicate_mapping.keys())
    foody_restaurants_with_duplicate_flag['befood_equivalent'] = foody_restaurants_with_duplicate_flag['restaurant_id'].map(duplicate_mapping)

    # Now create the main restaurant table
    restaurant_table = pd.concat([
        befood_restaurants[['restaurant_id', 'restaurant_name', 'latitude', 'longitude', 
                        'address', 'restaurant_rating', 'review_count', 
                        'city_id', 'district_id', 'food_type_id']], 
        foody_restaurants[['restaurant_id', 'restaurant_name', 'latitude', 'longitude', 
                        'address', 'restaurant_rating', 'review_count', 
                        'city_id', 'district_id']]
    ])

    # Fill missing food_type_id with a default value for foody restaurants
    restaurant_table['food_type_id'] = restaurant_table['food_type_id'].fillna(0).astype(int)

    # Create Dish table (already processed to handle duplicates)
    print("Creating Dish table...")
    dish_table = pd.concat([
        befood_dishes_copy[['dish_id', 'item_name', 'restaurant_id', 'category_id', 'category_name', 'price']],
        foody_dishes_copy[['dish_id', 'item_name', 'restaurant_id', 'category_id', 'category_name', 'price']]
    ])

    # Create User table
    print("Creating User table...")
    user_table = pd.concat([
        befood_reviews_copy[['user_id', 'user_name', 'platform_id']],
        foody_reviews_copy[['user_id', 'user_name', 'platform_id']]
    ]).drop_duplicates(subset=['user_id'])

    # Create Review table (already processed to handle duplicates)
    print("Creating Review table...")
    review_table = pd.concat([
        befood_reviews_copy[['rating_id', 'restaurant_id', 'user_id', 'rating', 'feedback', 'review_time']],
        foody_reviews_copy[['rating_id', 'restaurant_id', 'user_id', 'rating', 'feedback', 'review_time']]
    ])

    # Create Temp table with entries for both platforms for duplicate restaurants
    print("Creating Temp table...")
    # First, add befood entries
    temp_befood = pd.DataFrame()
    temp_befood['restaurant_id'] = befood_restaurants['restaurant_id']
    temp_befood['platform_id'] = 1  # Befood platform
    temp_befood['restaurant_rating'] = befood_restaurants['restaurant_rating']

    # Next, add foody entries for all restaurants (including duplicates)
    temp_foody = pd.DataFrame()
    temp_foody['restaurant_id'] = foody_restaurants['restaurant_id']
    temp_foody['platform_id'] = 2  # Foody platform
    temp_foody['restaurant_rating'] = foody_restaurants['restaurant_rating']

    # Add the mapping of duplicate ratings (these will be duplicate restaurants with their Befood IDs)
    temp_duplicates = pd.DataFrame(duplicate_ratings)

    # Combine all temp entries, excluding foody entries that will be replaced by duplicate_ratings
    temp_foody_filtered = temp_foody[~temp_foody['restaurant_id'].isin(duplicate_mapping.keys())]
    temp_table = pd.concat([temp_befood, temp_foody_filtered, temp_duplicates], ignore_index=True)
    temp_table['UniqueID'] = temp_table.index + 1  # Generate a unique ID
    temp_table = temp_table[['UniqueID', 'restaurant_id', 'platform_id', 'restaurant_rating']]

    # Create duplicate restaurant mapping file for reference
    print("Creating duplicate restaurant mapping file...")
    restaurant_id_mapping = {}
    for f_id, b_id in duplicate_mapping.items():
        if b_id:
            restaurant_id_mapping[f_id] = b_id
    mapping_df = pd.DataFrame(list(restaurant_id_mapping.items()), columns=['foody_id', 'befood_id'])

    # Write output files with utf-8 encoding
    print("Writing output files...")
    restaurant_table.to_csv(f'{output_dir}/Restaurant.csv', index=False, encoding='utf-8')
    city_table.to_csv(f'{output_dir}/City.csv', index=False, encoding='utf-8')
    district_table.to_csv(f'{output_dir}/District.csv', index=False, encoding='utf-8')
    platform_table.to_csv(f'{output_dir}/Platform.csv', index=False, encoding='utf-8')
    food_type.to_csv(f'{output_dir}/Food_type.csv', index=False, encoding='utf-8')
    dish_table.to_csv(f'{output_dir}/Dish.csv', index=False, encoding='utf-8')
    user_table.to_csv(f'{output_dir}/User.csv', index=False, encoding='utf-8')
    review_table.to_csv(f'{output_dir}/Review.csv', index=False, encoding='utf-8')
    temp_table.to_csv(f'{output_dir}/Temp.csv', index=False, encoding='utf-8')
    mapping_df.to_csv(f'{output_dir}/duplicate_restaurant_mapping.csv', index=False, encoding='utf-8')

    # Also save the foody restaurants with duplicate information for analysis
    foody_restaurants_with_duplicate_flag.to_csv(f'{output_dir}/foody_restaurants_with_duplicate_info.csv', index=False, encoding='utf-8')

    print("Initial data processing complete!")
    print(f"Found {valid_mappings} duplicate restaurants between platforms.")
    print(f"All data has been saved to {output_dir}/")
    
    return {
        'restaurant_table': restaurant_table,
        'food_type': food_type,
        'dish_table': dish_table,
        'review_table': review_table,
        'mapping_df': mapping_df
    } 

# ==================== STAGE 2: FIX REFERENCE ISSUES ====================

def fix_food_type_table(food_type_df=None):
    """
    Fix issues in Food_type table:
    1. Replace NULL with "Unknown" for food_type_id = 0
    2. Change food_type_id from 0 to 1 (checking if 1 already exists)
    """
    print("\n=== STAGE 2: FIX REFERENCE ISSUES ===")
    print("Fixing Food_type table...")
    
    # If food_type_df is not provided, read from file
    if food_type_df is None:
        food_type_df = pd.read_csv(os.path.join(DATA_PATHS['initial_output'], "Food_type.csv"))
    
    # Check if food_type_id = 0 exists and has NULL value
    mask = food_type_df['food_type_id'] == 0
    if mask.any():
        # Replace NULL with "Unknown" for food_type_id = 0
        food_type_df.loc[mask, 'food_type_name'] = "Unknown"
        
        # Check if food_type_id = 1 already exists
        if 1 in food_type_df['food_type_id'].values:
            # If exists, merge the records (keep the one with id=1)
            food_type_df = food_type_df[~mask]
        else:
            # Change food_type_id from 0 to 1
            food_type_df.loc[mask, 'food_type_id'] = 1
    
    # Save the fixed table
    food_type_df.to_csv(os.path.join(DATA_PATHS['fixed_references'], "Food_type.csv"), index=False)
    print("Food_type table fixed and saved.")
    return food_type_df

def update_restaurant_references(food_type_df, restaurant_df=None):
    """
    Update references to food_type_id=0 in Restaurant table to use food_type_id=1
    """
    print("Updating Restaurant references...")
    
    # If restaurant_df is not provided, read it from the original data
    if restaurant_df is None:
        restaurant_df = pd.read_csv(os.path.join(DATA_PATHS['initial_output'], "Restaurant.csv"))
    
    # Check if any restaurant has food_type_id=0
    if 0 in restaurant_df['food_type_id'].values:
        # Update references from 0 to 1
        restaurant_df.loc[restaurant_df['food_type_id'] == 0, 'food_type_id'] = 1
        print(f"Updated {sum(restaurant_df['food_type_id'] == 1)} restaurants with food_type_id=1")
    
    return restaurant_df

def handle_null_values_in_all_tables(food_type_df):
    """
    Check each table for null values in each column.
    Drop rows with null values as requested.
    Food_type is excluded because it's handled separately.
    """
    print("\n--- CHECKING NULL VALUES IN ALL TABLES ---")
    
    all_files = [f for f in os.listdir(DATA_PATHS['initial_output']) if f.endswith('.csv') and f != "Food_type.csv"]
    results = {}
    
    # Add the already processed Food_type table to the results
    results["Food_type"] = {
        'df': food_type_df,
        'original_rows': len(food_type_df),
        'rows_after_null_drop': len(food_type_df),
        'null_columns': {}
    }
    
    for file in all_files:
        table_name = file.replace('.csv', '')
        print(f"\nProcessing {table_name}...")
        file_path = os.path.join(DATA_PATHS['initial_output'], file)
        
        try:
            # Read the CSV file
            df = pd.read_csv(file_path)
            original_row_count = len(df)
            
            # Check for null values in each column
            null_counts = df.isnull().sum()
            columns_with_nulls = null_counts[null_counts > 0]
            
            if not columns_with_nulls.empty:
                print(f"Found null values in {table_name}:")
                for col, count in columns_with_nulls.items():
                    print(f"  - {col}: {count} null values")
                
                # Drop rows with null values as requested
                df = df.dropna()
                print(f"Dropped {original_row_count - len(df)} rows with null values")
            else:
                print(f"No null values found in {table_name}")
            
            # Save the processed table (without writing to output yet)
            # Will handle duplicates separately
            results[table_name] = {
                'df': df,
                'original_rows': original_row_count,
                'rows_after_null_drop': len(df),
                'null_columns': columns_with_nulls.to_dict()
            }
            
        except Exception as e:
            print(f"Error processing {file}: {str(e)}")
    
    return results

def handle_duplicate_primary_keys(table_results):
    """
    Handle duplicate primary keys in each table.
    Only drops duplicates based on the primary key column.
    """
    print("\n--- HANDLING DUPLICATE PRIMARY KEYS ---")
    
    # Map each table to its primary key column
    primary_keys = {
        "City": "city_id",
        "District": "district_id", 
        "Platform": "platform_id",
        "Review": "rating_id",
        "User": "user_id",
        "Dish": "dish_id",
        "Food_type": "food_type_id",
        "Restaurant": "restaurant_id",
        "Temp": "UniqueID"
    }
    
    duplicate_results = {}
    
    for table_name, result in table_results.items():
        if table_name not in primary_keys:
            print(f"No primary key defined for {table_name}, skipping duplicate check")
            # Save the dataframe without checking for duplicates
            result['df'].to_csv(os.path.join(DATA_PATHS['fixed_references'], f"{table_name}.csv"), index=False)
            continue
            
        primary_key = primary_keys[table_name]
        df = result['df']
        
        if primary_key not in df.columns:
            print(f"Primary key '{primary_key}' not found in {table_name}, skipping duplicate check")
            # Save the dataframe without checking for duplicates
            df.to_csv(os.path.join(DATA_PATHS['fixed_references'], f"{table_name}.csv"), index=False)
            continue
        
        # Count rows before dropping duplicates
        before_drop = len(df)
        
        # Find and count duplicate primary keys
        duplicates = df[df.duplicated(subset=[primary_key], keep=False)]
        duplicate_count = len(duplicates[primary_key].unique())
        
        if duplicate_count > 0:
            print(f"Found {duplicate_count} duplicate primary keys in {table_name}")
            duplicate_results[table_name] = duplicate_count
            
            # Drop duplicates, keeping only the first occurrence
            df = df.drop_duplicates(subset=[primary_key], keep='first')
            
            # Report how many rows were dropped
            removed_count = before_drop - len(df)
            print(f"Removed {removed_count} duplicate rows from {table_name}")
        else:
            print(f"No duplicate primary keys found in {table_name}")
            duplicate_results[table_name] = 0
        
        # Save the cleaned dataframe
        df.to_csv(os.path.join(DATA_PATHS['fixed_references'], f"{table_name}.csv"), index=False)
        
        # Update the result with final row count
        result['rows_after_duplicate_drop'] = len(df)
        result['duplicate_rows_dropped'] = before_drop - len(df)
        result['df'] = df  # Update the dataframe in results
    
    return duplicate_results

def verify_references(table_results):
    """
    Verify and fix references between tables after all processing
    """
    print("\n--- VERIFYING REFERENCES BETWEEN TABLES ---")
    
    # Make sure we have both Restaurant and Food_type in the results
    if 'Restaurant' in table_results and 'Food_type' in table_results:
        restaurant_df = table_results['Restaurant']['df']
        food_type_df = table_results['Food_type']['df']
        
        # Check for any remaining food_type_id=0 in Restaurant
        if 0 in restaurant_df['food_type_id'].values:
            print(f"Found {sum(restaurant_df['food_type_id'] == 0)} restaurants with food_type_id=0")
            
            # Update references from 0 to 1
            restaurant_df = update_restaurant_references(food_type_df, restaurant_df)
            
            # Save the updated dataframe
            restaurant_df.to_csv(os.path.join(DATA_PATHS['fixed_references'], "Restaurant.csv"), index=False)
            
            # Update the dataframe in results
            table_results['Restaurant']['df'] = restaurant_df
        else:
            print("No restaurants with food_type_id=0 found")
        
        # Check for any food_type_id in Restaurant that doesn't exist in Food_type
        invalid_food_types = set(restaurant_df['food_type_id'].unique()) - set(food_type_df['food_type_id'].unique())
        if invalid_food_types:
            print(f"Found {len(invalid_food_types)} invalid food_type_id references in Restaurant: {invalid_food_types}")
            
            # Set invalid food_type_id to 1
            for invalid_id in invalid_food_types:
                if invalid_id != 1:  # Skip if it's already 1
                    count = sum(restaurant_df['food_type_id'] == invalid_id)
                    print(f"Updating {count} restaurants with invalid food_type_id={invalid_id} to food_type_id=1")
                    restaurant_df.loc[restaurant_df['food_type_id'] == invalid_id, 'food_type_id'] = 1
            
            # Save the updated dataframe
            restaurant_df.to_csv(os.path.join(DATA_PATHS['fixed_references'], "Restaurant.csv"), index=False)
            
            # Update the dataframe in results
            table_results['Restaurant']['df'] = restaurant_df
        else:
            print("All food_type_id references in Restaurant are valid")
    else:
        print("Either Restaurant or Food_type table not found in results")
    
    return table_results

def generate_summary_report(table_results, duplicate_results):
    """Generate a summary report of all issues found and fixed"""
    report_path = os.path.join(DATA_PATHS['fixed_references'], "data_cleaning_report.txt")
    
    with open(report_path, "w") as f:
        f.write("=== DATA CLEANING SUMMARY REPORT ===\n\n")
        
        f.write("--- NULL VALUES AND DROPPED ROWS ---\n")
        for table_name, result in table_results.items():
            f.write(f"\n{table_name}:\n")
            f.write(f"  Original rows: {result['original_rows']}\n")
            f.write(f"  Rows after dropping nulls: {result['rows_after_null_drop']}\n")
            f.write(f"  Rows dropped due to nulls: {result['original_rows'] - result['rows_after_null_drop']}\n")
            
            if 'rows_after_duplicate_drop' in result:
                f.write(f"  Final rows after dropping duplicates: {result['rows_after_duplicate_drop']}\n")
                f.write(f"  Rows dropped due to duplicates: {result['duplicate_rows_dropped']}\n")
            
            if result['null_columns']:
                f.write("  Columns with null values:\n")
                for col, count in result['null_columns'].items():
                    f.write(f"    - {col}: {count} null values\n")
            else:
                f.write("  No columns with null values\n")
        
        f.write("\n--- DUPLICATE PRIMARY KEYS ---\n")
        for table, count in duplicate_results.items():
            if count > 0:
                f.write(f"\n{table}: {count} duplicate primary keys found and fixed\n")
    
    print(f"\nSummary report generated at {report_path}")
    
def fix_references():
    """Main function to fix reference issues"""
    # Fix Food_type table first - This replaces NULL with "Unknown" for food_type_id = 0
    # and changes food_type_id from 0 to 1
    food_type_df = fix_food_type_table()
    
    # Handle null values in all tables except Food_type
    table_results = handle_null_values_in_all_tables(food_type_df)
    
    # Handle duplicate primary keys
    duplicate_results = handle_duplicate_primary_keys(table_results)
    
    # Verify and fix references between tables
    table_results = verify_references(table_results)
    
    # Generate summary report
    generate_summary_report(table_results, duplicate_results)
    
    print("\nAll reference issues have been handled. Fixed data is in the output directory.")
    
    return table_results 

# ==================== STAGE 3: FIX RESTAURANT REFERENCES ====================

def load_restaurant_mapping():
    """Load the mapping between foody and befood restaurant IDs"""
    print("\n=== STAGE 3: FIX RESTAURANT REFERENCES ===")
    print("Loading restaurant mapping...")
    mapping_file = os.path.join(DATA_PATHS['fixed_references'], "duplicate_restaurant_mapping.csv")
    
    if not os.path.exists(mapping_file):
        print(f"Error: {mapping_file} not found")
        return {}
    
    mapping_df = pd.read_csv(mapping_file)
    
    # Create a dictionary mapping Foody IDs to Befood IDs
    restaurant_mapping = dict(zip(mapping_df['foody_id'], mapping_df['befood_id']))
    print(f"Loaded {len(restaurant_mapping)} restaurant ID mappings")
    
    return restaurant_mapping

def fix_dish_references(restaurant_mapping):
    """Fix restaurant references in the Dish table"""
    print("\nFixing restaurant references in Dish table...")
    dish_file = os.path.join(DATA_PATHS['fixed_references'], "Dish.csv")
    
    if not os.path.exists(dish_file):
        print(f"Error: {dish_file} not found")
        return None, {}
    
    # Load dish data
    print("Loading Dish data (this may take a while)...")
    start_time = time.time()
    dish_df = pd.read_csv(dish_file)
    print(f"Loaded {len(dish_df)} dishes in {time.time() - start_time:.2f} seconds")
    original_count = len(dish_df)
    
    # Get all unique restaurant IDs in the dish table
    unique_restaurant_ids = dish_df['restaurant_id'].unique()
    foody_ids_in_dishes = [r_id for r_id in unique_restaurant_ids if isinstance(r_id, str) and r_id.startswith('F_')]
    print(f"Found {len(foody_ids_in_dishes)} Foody restaurant IDs in Dish table")
    
    # Find how many Foody IDs need to be replaced
    foody_ids_to_replace = set(foody_ids_in_dishes).intersection(set(restaurant_mapping.keys()))
    print(f"Need to replace {len(foody_ids_to_replace)} Foody restaurant IDs with Befood IDs")
    
    # Track the replacements
    replacements = {}
    
    # Replace Foody IDs with Befood IDs
    start_time = time.time()
    for foody_id in foody_ids_to_replace:
        befood_id = restaurant_mapping[foody_id]
        mask = dish_df['restaurant_id'] == foody_id
        count = sum(mask)
        
        if count > 0:
            dish_df.loc[mask, 'restaurant_id'] = befood_id
            replacements[foody_id] = {
                'befood_id': befood_id,
                'count': count
            }
            print(f"  Replaced {foody_id} with {befood_id} ({count} dishes)")
    
    print(f"Replacement completed in {time.time() - start_time:.2f} seconds")
    
    # Save the updated dish data
    print("Saving updated Dish data...")
    start_time = time.time()
    output_file = os.path.join(DATA_PATHS['final_output'], "Dish.csv")
    dish_df.to_csv(output_file, index=False)
    print(f"Saved in {time.time() - start_time:.2f} seconds")
    
    print(f"Processed {original_count} dishes")
    print(f"Made changes to {sum(r['count'] for r in replacements.values())} dishes")
    print(f"Updated Dish table saved to {output_file}")
    
    return dish_df, replacements

def fix_review_references(restaurant_mapping):
    """Fix restaurant references in the Review table"""
    print("\nFixing restaurant references in Review table...")
    review_file = os.path.join(DATA_PATHS['fixed_references'], "Review.csv")
    
    if not os.path.exists(review_file):
        print(f"Error: {review_file} not found")
        return None, {}
    
    # Load review data
    print("Loading Review data (this may take a while)...")
    start_time = time.time()
    review_df = pd.read_csv(review_file)
    print(f"Loaded {len(review_df)} reviews in {time.time() - start_time:.2f} seconds")
    original_count = len(review_df)
    
    # Get all unique restaurant IDs in the review table
    unique_restaurant_ids = review_df['restaurant_id'].unique()
    foody_ids_in_reviews = [r_id for r_id in unique_restaurant_ids if isinstance(r_id, str) and r_id.startswith('F_')]
    print(f"Found {len(foody_ids_in_reviews)} Foody restaurant IDs in Review table")
    
    # Find how many Foody IDs need to be replaced
    foody_ids_to_replace = set(foody_ids_in_reviews).intersection(set(restaurant_mapping.keys()))
    print(f"Need to replace {len(foody_ids_to_replace)} Foody restaurant IDs with Befood IDs")
    
    # Track the replacements
    replacements = {}
    
    # Replace Foody IDs with Befood IDs
    start_time = time.time()
    for foody_id in foody_ids_to_replace:
        befood_id = restaurant_mapping[foody_id]
        mask = review_df['restaurant_id'] == foody_id
        count = sum(mask)
        
        if count > 0:
            review_df.loc[mask, 'restaurant_id'] = befood_id
            replacements[foody_id] = {
                'befood_id': befood_id,
                'count': count
            }
            print(f"  Replaced {foody_id} with {befood_id} ({count} reviews)")
    
    print(f"Replacement completed in {time.time() - start_time:.2f} seconds")
    
    # Save the updated review data
    print("Saving updated Review data...")
    start_time = time.time()
    output_file = os.path.join(DATA_PATHS['final_output'], "Review.csv")
    review_df.to_csv(output_file, index=False)
    print(f"Saved in {time.time() - start_time:.2f} seconds")
    
    print(f"Processed {original_count} reviews")
    print(f"Made changes to {sum(r['count'] for r in replacements.values())} reviews")
    print(f"Updated Review table saved to {output_file}")
    
    return review_df, replacements

def verify_restaurant_results(dish_df, review_df, restaurant_mapping):
    """Verify that all Foody IDs in the mapping have been replaced in Dish and Review tables"""
    print("\nVerifying results...")
    
    # Check dish table
    if dish_df is not None:
        dish_restaurant_ids = set(dish_df['restaurant_id'].unique())
        foody_ids_in_dishes = {r_id for r_id in dish_restaurant_ids if isinstance(r_id, str) and r_id.startswith('F_')}
        foody_ids_in_mapping = set(restaurant_mapping.keys())
        
        remaining_foody_ids_in_dishes = foody_ids_in_dishes.intersection(foody_ids_in_mapping)
        if remaining_foody_ids_in_dishes:
            print(f"Warning: Found {len(remaining_foody_ids_in_dishes)} Foody IDs in Dish table that weren't replaced")
            print(f"Example IDs: {list(remaining_foody_ids_in_dishes)[:5]}")
        else:
            print("All Foody IDs in Dish table that have Befood equivalents were successfully replaced")
    
    # Check review table
    if review_df is not None:
        review_restaurant_ids = set(review_df['restaurant_id'].unique())
        foody_ids_in_reviews = {r_id for r_id in review_restaurant_ids if isinstance(r_id, str) and r_id.startswith('F_')}
        
        remaining_foody_ids_in_reviews = foody_ids_in_reviews.intersection(foody_ids_in_mapping)
        if remaining_foody_ids_in_reviews:
            print(f"Warning: Found {len(remaining_foody_ids_in_reviews)} Foody IDs in Review table that weren't replaced")
            print(f"Example IDs: {list(remaining_foody_ids_in_reviews)[:5]}")
        else:
            print("All Foody IDs in Review table that have Befood equivalents were successfully replaced")

def copy_other_files():
    """Copy all other files from fixed_references directory to final_output directory"""
    print("\nCopying other files...")
    
    # List all files in fixed_references directory
    files = os.listdir(DATA_PATHS['fixed_references'])
    
    # Files we've already processed
    processed_files = ["Dish.csv", "Review.csv"]
    
    # Copy all other files
    for file in files:
        if file.endswith('.csv') and file not in processed_files:
            input_file = os.path.join(DATA_PATHS['fixed_references'], file)
            output_file = os.path.join(DATA_PATHS['final_output'], file)
            
            # Read and write the file to preserve encoding
            print(f"Copying {file}...")
            df = pd.read_csv(input_file)
            df.to_csv(output_file, index=False)
            print(f"Copied {file}")

def generate_restaurant_report(dish_replacements, review_replacements):
    """Generate a report summarizing the changes made to restaurant references"""
    print("\nGenerating report...")
    
    report_path = os.path.join(DATA_PATHS['final_output'], "restaurant_reference_fixes.txt")
    
    with open(report_path, "w") as f:
        f.write("=== RESTAURANT REFERENCE FIXES REPORT ===\n\n")
        
        f.write("--- DISH TABLE CHANGES ---\n")
        f.write(f"Total Foody IDs replaced: {len(dish_replacements)}\n")
        f.write(f"Total affected dishes: {sum(r['count'] for r in dish_replacements.values())}\n\n")
        
        if dish_replacements:
            f.write("Top 10 replacements by number of dishes affected:\n")
            sorted_replacements = sorted(dish_replacements.items(), key=lambda x: x[1]['count'], reverse=True)
            for foody_id, info in sorted_replacements[:10]:
                f.write(f"  {foody_id} -> {info['befood_id']}: {info['count']} dishes\n")
        
        f.write("\n--- REVIEW TABLE CHANGES ---\n")
        f.write(f"Total Foody IDs replaced: {len(review_replacements)}\n")
        f.write(f"Total affected reviews: {sum(r['count'] for r in review_replacements.values())}\n\n")
        
        if review_replacements:
            f.write("Top 10 replacements by number of reviews affected:\n")
            sorted_replacements = sorted(review_replacements.items(), key=lambda x: x[1]['count'], reverse=True)
            for foody_id, info in sorted_replacements[:10]:
                f.write(f"  {foody_id} -> {info['befood_id']}: {info['count']} reviews\n")
    
    print(f"Report saved to {report_path}")

def fix_restaurant_refs():
    """Main function to fix restaurant references"""
    # Load the restaurant mapping from the existing file
    restaurant_mapping = load_restaurant_mapping()
    
    # Fix references in Dish table
    dish_df, dish_replacements = fix_dish_references(restaurant_mapping)
    
    # Fix references in Review table
    review_df, review_replacements = fix_review_references(restaurant_mapping)
    
    # Verify the results
    verify_restaurant_results(dish_df, review_df, restaurant_mapping)
    
    # Copy other files
    copy_other_files()
    
    # Generate report
    generate_restaurant_report(dish_replacements, review_replacements)
    
    print("\nRestaurant reference fixes complete!")
    
    return dish_df, review_df 

# ==================== STAGE 4: FIX DISH IDs ====================

def fix_dish_ids(dish_df=None):
    """Remove .0 suffix from dish_id and category_id values in Dish.csv"""
    print("\n=== STAGE 4: FIX DISH IDs ===")
    
    # If dish_df is not provided, read it from the file
    if dish_df is None:
        dish_file = os.path.join(DATA_PATHS['final_output'], "Dish.csv")
        if not os.path.exists(dish_file):
            print(f"Error: {dish_file} not found")
            return None
            
        # Load dish data
        print("Loading Dish data (this may take a while)...")
        start_time = time.time()
        dish_df = pd.read_csv(dish_file)
        print(f"Loaded {len(dish_df)} dishes in {time.time() - start_time:.2f} seconds")
    
    # Fix dish_id column
    print("\n--- Fixing dish_id column ---")
    
    # Check the dish_id column
    print("Sample dish_id values before fixing:")
    print(dish_df['dish_id'].head(10).tolist())
    
    # Count how many dish_ids need fixing
    has_decimal = dish_df['dish_id'].astype(str).str.contains(r'\.0$')
    num_to_fix = has_decimal.sum()
    print(f"Found {num_to_fix} dish_ids with '.0' suffix that need fixing")
    
    if num_to_fix > 0:
        # Clean the dish_id values by removing the .0 suffix
        print("Removing '.0' suffix from dish_ids...")
        start_time = time.time()
        
        # Convert to string and remove .0 suffix
        dish_df['dish_id'] = dish_df['dish_id'].astype(str).str.replace(r'\.0$', '', regex=True)
        
        print(f"Cleaned dish_ids in {time.time() - start_time:.2f} seconds")
        
        # Show sample of fixed dish_ids
        print("Sample dish_id values after fixing:")
        print(dish_df['dish_id'].head(10).tolist())
    else:
        print("No dish_ids need fixing")
    
    # Fix category_id column
    print("\n--- Fixing category_id column ---")
    
    # Check if category_id column exists
    if 'category_id' in dish_df.columns:
        # Check the category_id column
        print("Sample category_id values before fixing:")
        print(dish_df['category_id'].head(10).tolist())
        
        # Count how many category_ids need fixing
        has_decimal = dish_df['category_id'].astype(str).str.contains(r'\.0$')
        num_to_fix = has_decimal.sum()
        print(f"Found {num_to_fix} category_ids with '.0' suffix that need fixing")
        
        if num_to_fix > 0:
            # Clean the category_id values by removing the .0 suffix
            print("Removing '.0' suffix from category_ids...")
            start_time = time.time()
            
            # Convert to string and remove .0 suffix
            dish_df['category_id'] = dish_df['category_id'].astype(str).str.replace(r'\.0$', '', regex=True)
            
            print(f"Cleaned category_ids in {time.time() - start_time:.2f} seconds")
            
            # Show sample of fixed category_ids
            print("Sample category_id values after fixing:")
            print(dish_df['category_id'].head(10).tolist())
        else:
            print("No category_ids need fixing")
    else:
        print("No category_id column found in Dish.csv")
    
    # Save the updated dish data
    print("\nSaving updated Dish data...")
    dish_file = os.path.join(DATA_PATHS['final_output'], "Dish.csv")
    start_time = time.time()
    dish_df.to_csv(dish_file, index=False)
    print(f"Saved in {time.time() - start_time:.2f} seconds")
    
    print(f"Updated Dish table saved to {dish_file}")
    
    return dish_df

# ==================== STAGE 5: CLEAN DISH DATA ====================

def clean_dish_data(dish_df=None):
    """Clean dish data by removing entries with invalid restaurant_id references"""
    print("\n=== STAGE 5: CLEAN DISH DATA ===")
    
    # If dish_df is not provided, read it from the file
    if dish_df is None:
        dish_file = os.path.join(DATA_PATHS['final_output'], "Dish.csv")
        if not os.path.exists(dish_file):
            print(f"Error: {dish_file} not found")
            return None
            
        print("Loading Dish data...")
        dish_df = pd.read_csv(dish_file)
    
    # Read the Restaurant.csv file to get list of valid restaurant_ids
    restaurant_file = os.path.join(DATA_PATHS['final_output'], "Restaurant.csv")
    if not os.path.exists(restaurant_file):
        print(f"Error: {restaurant_file} not found")
        return dish_df
        
    print("Loading Restaurant data...")
    restaurant_df = pd.read_csv(restaurant_file)
    
    # Get list of valid restaurant_ids from Restaurant.csv
    valid_restaurant_ids = set(restaurant_df['restaurant_id'].astype(str))
    
    # Count initial rows
    initial_count = len(dish_df)
    
    # Filter Dish.csv to keep only rows with restaurant_id in Restaurant.csv
    dish_df['restaurant_id'] = dish_df['restaurant_id'].astype(str)
    dish_df = dish_df[dish_df['restaurant_id'].isin(valid_restaurant_ids)]
    
    # Count removed rows
    removed_count = initial_count - len(dish_df)
    
    # Save the cleaned dataframe back to Dish.csv
    output_file = os.path.join(DATA_PATHS['validated_data'], "Dish.csv")
    dish_df.to_csv(output_file, index=False)
    
    print(f"Processing completed successfully:")
    print(f"- Initial rows in Dish.csv: {initial_count}")
    print(f"- Rows removed: {removed_count}")
    print(f"- Remaining rows: {len(dish_df)}")
    print(f"- Cleaned file saved as '{output_file}'")
    
    return dish_df 

# ==================== STAGE 6: VALIDATE DATA ====================

def check_null_values():
    """Check for NULL values in user_name column in validated data files"""
    print("\n=== STAGE 6: VALIDATE DATA ===")
    print("\n--- Checking for NULL values in user_name ---")
    
    # Load User data
    start_time = time.time()
    print("Loading User.csv...")
    user_file = os.path.join(DATA_PATHS['validated_data'], 'User.csv')
    
    if not os.path.exists(user_file):
        print(f"Error: {user_file} not found")
        return 0, 0
        
    user_df = pd.read_csv(user_file)
    print(f"User.csv loaded in {time.time() - start_time:.2f} seconds")
    
    # Check for NULL values in user_name
    null_count = user_df['user_name'].isna().sum()
    empty_count = (user_df['user_name'] == '').sum()
    
    print(f"\nTotal users: {len(user_df):,}")
    print(f"NULL values in user_name: {null_count:,}")
    print(f"Empty strings in user_name: {empty_count:,}")
    
    if null_count > 0 or empty_count > 0:
        print("\nExample users with missing names:")
        if null_count > 0:
            print(user_df[user_df['user_name'].isna()].head(5))
        if empty_count > 0:
            print(user_df[user_df['user_name'] == ''].head(5))
    else:
        print("\nSuccess: No NULL or empty user_name values found!")
    
    return null_count, empty_count

def verify_user_ids_match():
    """Verify that user_ids in Review.csv exist in User.csv"""
    print("\n--- Checking user_id matching between User.csv and Review.csv ---")
    
    # Load User data
    start_time = time.time()
    print("Loading User.csv...")
    user_file = os.path.join(DATA_PATHS['validated_data'], 'User.csv')
    
    if not os.path.exists(user_file):
        print(f"Error: {user_file} not found")
        return set(), set()
        
    user_df = pd.read_csv(user_file)
    user_ids = set(user_df['user_id'])
    print(f"User.csv loaded in {time.time() - start_time:.2f} seconds")
    print(f"Unique user_ids in User.csv: {len(user_ids):,}")
    
    # Load Review data
    start_time = time.time()
    print("\nLoading Review.csv...")
    review_file = os.path.join(DATA_PATHS['validated_data'], 'Review.csv')
    
    if not os.path.exists(review_file):
        print(f"Error: {review_file} not found")
        return set(), set()
        
    review_df = pd.read_csv(review_file)
    review_user_ids = set(review_df['user_id'].unique())
    print(f"Review.csv loaded in {time.time() - start_time:.2f} seconds")
    print(f"Unique user_ids in Review.csv: {len(review_user_ids):,}")
    
    # Find user_ids in Review but not in User
    missing_user_ids = review_user_ids - user_ids
    
    # Find user_ids in User but not used in any Review
    unused_user_ids = user_ids - review_user_ids
    
    print(f"\nUser_ids in Review but not in User table: {len(missing_user_ids):,}")
    if missing_user_ids:
        print(f"Sample missing user_ids: {list(missing_user_ids)[:5]}")
    
    print(f"User_ids in User table not used in any Review: {len(unused_user_ids):,}")
    if len(unused_user_ids) > 0 and len(unused_user_ids) < 10:
        print(f"Unused user_ids: {list(unused_user_ids)}")
    elif len(unused_user_ids) >= 10:
        print(f"Sample unused user_ids: {list(unused_user_ids)[:5]}")
    
    # Final verification
    if not missing_user_ids:
        print("\n✅ SUCCESS: All user_ids in Review.csv exist in User.csv!")
    else:
        print("\n❌ ERROR: Some user_ids in Review.csv don't exist in User.csv!")
    
    return missing_user_ids, unused_user_ids

def validate_data():
    """Run all data validation checks"""
    # Copy remaining files from final_output to validated_data if they don't exist yet
    for file in os.listdir(DATA_PATHS['final_output']):
        if file.endswith('.csv'):
            source_file = os.path.join(DATA_PATHS['final_output'], file)
            target_file = os.path.join(DATA_PATHS['validated_data'], file)
            
            if not os.path.exists(target_file):
                print(f"Copying {file} to validated data directory...")
                df = pd.read_csv(source_file)
                df.to_csv(target_file, index=False)
    
    # Check for NULL values in user_name
    null_count, empty_count = check_null_values()
    
    # Verify user_ids match between User.csv and Review.csv
    missing_user_ids, unused_user_ids = verify_user_ids_match()
    
    # Summary
    print("\n=== VALIDATION SUMMARY ===")
    if null_count == 0 and empty_count == 0 and len(missing_user_ids) == 0:
        print("✅ All validation checks passed! Data is ready for import.")
    else:
        print("❌ Issues found in the data:")
        if null_count > 0 or empty_count > 0:
            print(f"  - {null_count + empty_count} users with missing names")
        if missing_user_ids:
            print(f"  - {len(missing_user_ids)} user_ids in Review that don't exist in User table")
    
    return {
        'null_count': null_count,
        'empty_count': empty_count,
        'missing_user_ids': missing_user_ids,
        'unused_user_ids': unused_user_ids
    }

# ==================== STAGE 7: PROCESS SENTIMENT LABELS ====================

def process_sentiment_labels():
    """Process sentiment labels for feedback"""
    print("\n=== STAGE 7: PROCESS SENTIMENT LABELS ===")
    
    # Set paths
    befood_path = DATA_PATHS['befood_sentiment']
    foody_path = DATA_PATHS['foody_sentiment']
    review_path = os.path.join(DATA_PATHS['validated_data'], "Review.csv")
    output_path = os.path.join(DATA_PATHS['validated_data'], "Feedback_label.csv")
    rejected_path = os.path.join(DATA_PATHS['validated_data'], "rejected_feedback_label.csv")
    
    # Check if sentiment files exist
    if not os.path.exists(befood_path) or not os.path.exists(foody_path):
        print("Error: Sentiment data files not found. Skipping sentiment processing.")
        return None
    
    # Read files with the correct encoding
    print("Reading Befood sentiment data...")
    befood_df = pd.read_csv(befood_path)
    print(f"Befood data shape: {befood_df.shape}")

    print("Reading Foody sentiment data...")
    foody_df = pd.read_csv(foody_path)
    print(f"Foody data shape: {foody_df.shape}")

    # Add prefixes to rating_id
    print("Adding prefixes to rating_id...")
    befood_df['rating_id'] = 'BR_' + befood_df['rating_id'].astype(str)
    foody_df['rating_id'] = 'FR_' + foody_df['rating_id'].astype(str)

    # Merge the dataframes
    print("Merging dataframes...")
    merged_df = pd.concat([befood_df, foody_df], ignore_index=True)
    print(f"Merged data shape: {merged_df.shape}")

    # Load review data to get valid rating_ids
    print("Reading Review data to get valid rating_ids...")
    # Define chunksize to handle large file
    chunksize = 500000
    rating_ids = set()

    # Read the Review file in chunks
    for chunk in pd.read_csv(review_path, chunksize=chunksize):
        rating_ids.update(chunk['rating_id'].astype(str).unique())
        
    print(f"Total unique rating_ids in Review: {len(rating_ids)}")

    # Filter merged data to match Feedback_label structure
    # feedback_label_id, label, rating_label, rating_id
    print("Restructuring data to match Feedback_label...")

    # Rename columns to match Feedback_label table
    # - feedback_id becomes feedback_label_id
    # - category becomes label
    # - re_rating becomes rating_label
    # - rating_id remains rating_id
    result_df = merged_df.rename(columns={
        'feedback_id': 'feedback_label_id',
        'category': 'label',
        're_rating': 'rating_label'
    })

    # Keep only columns that are in Feedback_label table
    result_df = result_df[['feedback_label_id', 'label', 'rating_label', 'rating_id']]

    # Convert data types to match the Feedback_label table
    result_df['feedback_label_id'] = result_df['feedback_label_id'].astype(int)
    result_df['rating_label'] = result_df['rating_label'].astype(float)

    # Filter to keep only rows where rating_id exists in Review
    print("Filtering for existing rating_ids...")
    result_df['exists_in_review'] = result_df['rating_id'].isin(rating_ids)

    # Split into accepted and rejected dataframes
    accepted_df = result_df[result_df['exists_in_review'] == True].drop(columns=['exists_in_review'])
    rejected_df = result_df[result_df['exists_in_review'] == False].drop(columns=['exists_in_review'])

    print(f"Accepted rows: {accepted_df.shape[0]}")
    print(f"Rejected rows: {rejected_df.shape[0]}")

    # Save the results
    print(f"Saving results to {output_path} and {rejected_path}")
    accepted_df.to_csv(output_path, index=False)
    rejected_df.to_csv(rejected_path, index=False)

    print("Sentiment processing complete!")
    
    return accepted_df 

# ==================== MAIN EXECUTION FUNCTION ====================

def main():
    """
    Execute the complete data processing pipeline:
    1. Initial data processing
    2. Fix reference issues
    3. Fix restaurant references
    4. Fix dish IDs
    5. Clean dish data
    6. Validate data
    7. Process sentiment labels
    """
    print("\n=== STARTING FULL PIPELINE EXECUTION ===")
    start_time = time.time()
    
    # Stage 1: Initial data processing
    stage1_start = time.time()
    initial_results = process_initial_data()
    print(f"Stage 1 completed in {time.time() - stage1_start:.2f} seconds")
    
    # Stage 2: Fix reference issues
    stage2_start = time.time()
    fixed_tables = fix_references()
    print(f"Stage 2 completed in {time.time() - stage2_start:.2f} seconds")
    
    # Stage 3: Fix restaurant references
    stage3_start = time.time()
    dish_df, review_df = fix_restaurant_refs()
    print(f"Stage 3 completed in {time.time() - stage3_start:.2f} seconds")
    
    # Stage 4: Fix dish IDs
    stage4_start = time.time()
    dish_df = fix_dish_ids(dish_df)
    print(f"Stage 4 completed in {time.time() - stage4_start:.2f} seconds")
    
    # Stage 5: Clean dish data
    stage5_start = time.time()
    dish_df = clean_dish_data(dish_df)
    print(f"Stage 5 completed in {time.time() - stage5_start:.2f} seconds")
    
    # Stage 6: Validate data
    stage6_start = time.time()
    validation_results = validate_data()
    print(f"Stage 6 completed in {time.time() - stage6_start:.2f} seconds")
    
    # Stage 7: Process sentiment labels
    stage7_start = time.time()
    feedback_df = process_sentiment_labels()
    print(f"Stage 7 completed in {time.time() - stage7_start:.2f} seconds")
    
    # Generate final report
    total_time = time.time() - start_time
    print("\n=== PIPELINE EXECUTION SUMMARY ===")
    print(f"Total execution time: {total_time:.2f} seconds ({total_time/60:.2f} minutes)")
    print("\nFinal data tables created:")
    for file in os.listdir(DATA_PATHS['validated_data']):
        if file.endswith('.csv'):
            file_path = os.path.join(DATA_PATHS['validated_data'], file)
            file_size = os.path.getsize(file_path) / (1024 * 1024)  # Size in MB
            
            # Count rows in the file
            try:
                row_count = sum(1 for _ in open(file_path)) - 1  # Subtract 1 for header
            except:
                row_count = "Unknown"
                
            print(f"- {file}: {file_size:.2f} MB, {row_count:,} rows")
    
    print("\nPipeline execution completed successfully!")
    print(f"Final data is available in: {DATA_PATHS['validated_data']}")

if __name__ == "__main__":
    main() 