# Restaurant Review Data Processing Pipeline
**Note: Due to security reasons, we can't upload the full dataset at the moment. This issue will be addressed in the future.**
## Directory Structure

This project processes restaurant review data from Befood and Foody platforms, transforming raw data into structured database tables.
```
analysis_data/
└── processing_data/
    ├── raw_data/             # Raw input data
    │   ├── befood/           # Raw data from Befood platform
    │   └── foody/            # Raw data from Foody platform
    ├── cleaned_data/         # Initial cleaned data
    │   ├── befood_restaurants_cleaned.csv
    │   ├── foody_restaurants_cleaned.csv
    │   ├── befood_dishes_cleaned.csv
    │   ├── foody_dishes_cleaned.csv
    │   ├── befood_reviews_cleaned.csv
    │   └── foody_reviews_cleaned.csv
    ├── output/               # Final processed output data
    │   ├── Restaurant.csv
    │   ├── Dish.csv
    │   ├── Review.csv
    │   ├── User.csv
    │   ├── City.csv
    │   ├── District.csv
    │   ├── Platform.csv
    │   ├── Food_type.csv
    │   ├── Temp.csv
    │   ├── Feedback_label.csv
    │   └── [various report files]
    ├── sentiment_category/   # Sentiment analysis data
    │   ├── befood_predicted_sentiment.csv
    │   └── foody_predicted_sentiment.csv
    └── EDA.ipynb                     # Exploratory data analysis
    └── README.md
    └── restaurant_data_pipeline.py  # Main processing script
```

### /raw_data
Contains the original raw data from both platforms:
- **befood/**: Raw data files from Befood platform, including restaurants, dishes, and reviews
- **foody/**: Raw data files from Foody platform, including restaurants, dishes, and reviews

### /cleaned_data
Contains the initially cleaned data created by running `clean_data.ipynb`:
- **befood_restaurants_cleaned.csv**: Preprocessed restaurant data from Befood
- **foody_restaurants_cleaned.csv**: Preprocessed restaurant data from Foody
- **befood_dishes_cleaned.csv**: Preprocessed dish/menu data from Befood
- **foody_dishes_cleaned.csv**: Preprocessed dish/menu data from Foody
- **befood_reviews_cleaned.csv**: Preprocessed review data from Befood
- **foody_reviews_cleaned.csv**: Preprocessed review data from Foody
- **clean_data.ipynb**: Jupyter notebook with initial cleaning process

### /output
Contains the final processed data tables that match the ER diagram schema:
- **Restaurant.csv**: Combined restaurant data from both platforms
- **Dish.csv**: Menu items with standardized IDs and references
- **Review.csv**: User reviews with standardized IDs and references
- **User.csv**: Unique users across both platforms
- **City.csv**: City reference table
- **District.csv**: District reference table
- **Platform.csv**: Platform reference table (Befood=1, Foody=2)
- **Food_type.csv**: Restaurant categories/cuisines
- **Temp.csv**: Table with platform-specific restaurant ratings
- **Feedback_label.csv**: Sentiment analysis labels for reviews
- **ER_Diagram.png**: Database schema diagram

### /sentiment_category
Contains sentiment analysis data used to generate Feedback_label.csv:
- **befood_predicted_sentiment.csv**: Sentiment predictions for Befood reviews
- **foody_predicted_sentiment.csv**: Sentiment predictions for Foody reviews

## Main Processing Script: restaurant_data_pipeline.py

This script performs the complete data processing pipeline in 7 sequential stages.

### How to Use

1. Ensure all required directories exist (raw_data, cleaned_data, output, sentiment_category)
2. Make sure you have already run clean_data.ipynb to create the files in /cleaned_data
3. Run the script:
   ```
   python restaurant_data_pipeline.py
   ```
4. The script will process all stages and save results to the /output directory

### Detailed Pipeline Stages

1. **Initial Data Processing**
   - Loads data from /cleaned_data
   - Standardizes city names and addresses
   - Creates mapping between duplicate restaurants across platforms
   - Generates initial versions of all database tables

2. **Fix Reference Issues**
   - Fixes Food_type table (replaces NULL with "Unknown")
   - Updates restaurant references to food types
   - Handles NULL values in all tables
   - Removes duplicate primary keys

3. **Fix Restaurant References**
   - Maps Foody restaurant IDs to their Befood equivalents
   - Updates references in Dish and Review tables
   - Generates a report of all changes made

4. **Fix Dish IDs**
   - Removes decimal suffixes (.0) from dish_id and category_id
   - Ensures consistent ID formats

5. **Clean Dish Data**
   - Removes dishes with invalid restaurant references
   - Ensures referential integrity

6. **Validate Data**
   - Checks for NULL values in user_name
   - Verifies user_id references between Review and User tables
   - Produces validation report

7. **Process Sentiment Labels**
   - Processes sentiment analysis results
   - Creates Feedback_label table
   - Filters out invalid references

### Configuration

You can modify the input/output paths in the `DATA_PATHS` dictionary at the top of the script:

```python
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
```

### Requirements

- Python 3.9+
- pandas
- numpy
- re (regular expressions)
- seaborn
- matplotlib

## Workflow Example

1. Start with raw data in /raw_data
2. Run clean_data.ipynb to create files in /cleaned_data
3. Run restaurant_data_pipeline.py to execute the full pipeline
4. Check the final results in /output
5. Verify data integrity with the validation reports

## Notes
- The script handles various edge cases like encoding issues, missing values, and format inconsistencies
- Restaurant deduplication is performed to map equivalent restaurants between platforms
- The final tables conform to the database schema shown in ER_Diagram.png 
![alt text]([https://github.com/adam-p/markdown-here/raw/master/src/common/images/icon48.png](https://github.com/Team-7-Grab-Tech-Bootcamp-2025/data-analysis/blob/main/output/ER_Diagram.png) "ER Diagram")
