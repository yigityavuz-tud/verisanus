import os
import yaml
import pandas as pd
import hashlib
from datetime import datetime
from apify_client import ApifyClient
from pathlib import Path
# Import from translation_utils instead
from translation_utils import translate_column, log_deepl_usage, load_config

def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def get_apify_token():
    config = load_config()['google_maps']
    token_file = config['api_settings']['token_file']
    try:
        with open(token_file, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        raise ValueError(f"Token file {token_file} not found. Please check the config file.")

# Remove duplicate translation-related functions that are now in translation_utils.py
# Delete these functions:
# - get_deepl_token
# - get_deepseek_token
# - count_characters
# - log_deepl_usage
# - translate_with_deepseek

def translate_owner_responses(df):
    """
    Translate non-English owner responses to English using the configured translation service.
    Only translates rows where responseFromOwnerText is not empty and originalLanguage is not 'en'.
    Returns the translated DataFrame and the number of characters translated.
    """
    config = load_config()['google_maps']
    translation_service = config['api_settings'].get('translation_service', 'deepl')
    
    # Create new column for English translations if it doesn't exist
    if 'responseFromOwnerText_en' not in df.columns:
        df['responseFromOwnerText_en'] = df['responseFromOwnerText']
    
    # Get rows that need translation
    to_translate = df[
        (df['responseFromOwnerText'].notna()) & 
        (df['responseFromOwnerText'] != '') & 
        (df['originalLanguage'] != 'en')
    ]
    
    characters_translated = 0
    
    if not to_translate.empty:
        print(f"Translating {len(to_translate)} owner responses using {translation_service}...")
        translated_df, characters = translate_column(
            to_translate, 
            'responseFromOwnerText', 
            'responseFromOwnerText_en',
            translation_service=translation_service
        )
        
        # Update the original dataframe with translations
        df.loc[to_translate.index, 'responseFromOwnerText_en'] = translated_df['responseFromOwnerText_en']
        characters_translated = characters
    
    return df, characters_translated

def make_timezone_naive(df, datetime_columns):
    """Convert datetime columns to timezone-naive"""
    for col in datetime_columns:
        if col in df.columns:
            # First convert to datetime if not already
            df[col] = pd.to_datetime(df[col], errors='coerce')
            # Then remove timezone if present
            if df[col].dt.tz is not None:
                df[col] = df[col].dt.tz_localize(None)
    return df

def get_establishments_to_scrape():
    config = load_config()['google_maps']
    df = pd.read_excel('establishments/establishment_base.xlsx')
    
    # Check if custom_place_ids is specified and not empty
    if 'custom_place_ids' in config and config['custom_place_ids']:
        print("Custom place IDs found, overriding scraping criteria...")
        # Filter only the specified place IDs
        df = df[df['placeId'].isin(config['custom_place_ids'])]
        return df
    
    # Convert googleMapsScrapedAt to datetime, handling empty values
    df['googleMapsScrapedAt'] = pd.to_datetime(df['googleMapsScrapedAt'], errors='coerce')
    if df['googleMapsScrapedAt'].dt.tz is not None:
        df['googleMapsScrapedAt'] = df['googleMapsScrapedAt'].dt.tz_localize(None)
    
    # Apply filtering criteria
    if config['scraping_criteria']['scrape_unscraped']:
        # If we're scraping unscraped, we don't need to check the date
        df = df[df['googleMapsScrapedAt'].isna()]
    elif config['scraping_criteria']['scrape_before_date']:
        # Only check date if we're not specifically looking for unscraped
        scrape_before_date = pd.to_datetime(config['scraping_criteria']['scrape_before_date'])
        if scrape_before_date.tz is not None:
            scrape_before_date = scrape_before_date.tz_localize(None)
        df = df[df['googleMapsScrapedAt'] < scrape_before_date]
    
    # Filter by review count
    df = df[(df['reviewsCount'] >= config['scraping_criteria']['min_reviews_per_establishment']) &
            (df['reviewsCount'] <= config['scraping_criteria']['max_reviews_per_establishment'])]
    
    # Sort by review count and limit total reviews
    df = df.sort_values('reviewsCount', ascending=False)
    total_reviews = 0
    establishments_to_scrape = []
    
    for _, row in df.iterrows():
        if total_reviews + row['reviewsCount'] <= config['scraping_criteria']['max_total_reviews']:
            establishments_to_scrape.append(row)
            total_reviews += row['reviewsCount']
        else:
            break
    
    return pd.DataFrame(establishments_to_scrape)

def generate_review_id(place_id, published_at):
    return hashlib.md5(f"{place_id}_{published_at}".encode()).hexdigest()

def scrape_reviews(establishments):
    config = load_config()['google_maps']
    token = get_apify_token()
    client = ApifyClient(token)
    
    all_reviews = []
    for _, establishment in establishments.iterrows():
        run_input = {
            "startUrls": [{"url": establishment['url']}],
            "maxReviews": config['api_settings']['max_reviews'],
            "reviewsSort": config['api_settings']['reviews_sort'],
            "language": config['api_settings']['language'],
            "reviewsOrigin": config['api_settings']['reviews_origin'],
            "personalData": config['api_settings']['personal_data'],
        }
        
        run = client.actor("Xb8osYTtOjlsgI6k9").call(run_input=run_input)
        
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            # Generate reviewId during scraping
            item['reviewId'] = generate_review_id(item['placeId'], item['publishedAtDate'])
            all_reviews.append(item)
    
    reviews_df = pd.DataFrame(all_reviews)
    # Make datetime columns timezone-naive
    datetime_columns = ['publishedAtDate', 'updatedAtDate']
    reviews_df = make_timezone_naive(reviews_df, datetime_columns)
    return reviews_df

def save_reviews(reviews_df):
    # Define the columns we want to keep
    required_columns = [
        'reviewId',
        'reviewerNumberOfReviews',
        'isLocalGuide',
        'text',
        'textTranslated',
        'publishedAtDate',
        'likesCount',
        'stars',
        'responseFromOwnerDate',
        'responseFromOwnerText',
        'originalLanguage',
        'translatedLanguage',
        'placeId',
        'scrapedAt',
        'language'
    ]
    
    # Convert publishedAtDate to date format
    reviews_df['publishedAtDate'] = pd.to_datetime(reviews_df['publishedAtDate'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')
    reviews_df['publishedAtDate'] = reviews_df['publishedAtDate'].dt.date
    
    # Keep only the required columns
    reviews_df = reviews_df[required_columns]
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_file = f"reviews/google/googleReviews_{timestamp}.xlsx"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the reviews
    reviews_df.to_excel(output_file, index=False)
    print(f"Reviews saved to {output_file}")

def unify_reviews():
    google_dir = Path("reviews/google")
    all_reviews = []
    
    # Define the columns we want to keep
    required_columns = [
        'reviewerNumberOfReviews',
        'isLocalGuide',
        'text',
        'textTranslated',
        'publishedAtDate',
        'likesCount',
        'stars',
        'responseFromOwnerDate',
        'responseFromOwnerText',
        'responseFromOwnerText_en',
        'originalLanguage',
        'translatedLanguage',
        'placeId',
        'scrapedAt',
        'language',
        'reviewId',
        'sourceFile'
    ]
    
    # First, try to find the most recent unified file
    unified_files = list(google_dir.glob("allGoogleReviews_*.xlsx"))
    
    if unified_files:
        # Case 1: Existing allGoogleReviews_ file found
        latest_unified = max(unified_files)
        print(f"Found existing unified file: {latest_unified}")
        existing_df = pd.read_excel(latest_unified)
        
        # Drop rows where sourceFile is empty/null
        existing_df = existing_df.dropna(subset=['sourceFile'])
        
        # Get list of already processed source files (using just the filename)
        processed_files = set(Path(f).name for f in existing_df['sourceFile'].unique())
        print(f"Previously processed files: {len(processed_files)}")
        
        # Add existing reviews to all_reviews
        all_reviews.append(existing_df)
        
        # Process only new files
        new_files = 0
        for file in google_dir.glob("googleReviews_*.xlsx"):
            if file.name.startswith("allGoogleReviews_"):
                continue  # Skip unified files
                
            # Only process files that haven't been processed before
            if file.name not in processed_files:
                print(f"Processing new file: {file}")
                df = pd.read_excel(file)
                new_files += 1
                
                # Add sourceFile column if it doesn't exist
                if 'sourceFile' not in df.columns:
                    df['sourceFile'] = str(file)
                
                # Generate reviewId if it doesn't exist
                if 'reviewId' not in df.columns:
                    df['reviewId'] = df.apply(
                        lambda row: generate_review_id(row['placeId'], row['publishedAtDate']), 
                        axis=1
                    )
                
                # Keep only the required columns
                for col in required_columns:
                    if col not in df.columns:
                        df[col] = None
                
                df = df[required_columns]
                all_reviews.append(df)
                
        print(f"Processed {new_files} new files")
    else:
        # Case 2: No existing allGoogleReviews_ file
        print("No existing unified file found. Processing all files...")
        for file in google_dir.glob("googleReviews_*.xlsx"):
            if file.name.startswith("allGoogleReviews_"):
                continue
            
            print(f"Processing file: {file}")
            df = pd.read_excel(file)
            
            # Add sourceFile column
            df['sourceFile'] = str(file)
            
            # Generate reviewId if it doesn't exist
            if 'reviewId' not in df.columns:
                df['reviewId'] = df.apply(
                    lambda row: generate_review_id(row['placeId'], row['publishedAtDate']), 
                    axis=1
                )
            
            # Keep only the required columns
            for col in required_columns:
                if col not in df.columns:
                    df[col] = None
            
            df = df[required_columns]
            all_reviews.append(df)
    
    if all_reviews:
        unified_df = pd.concat(all_reviews, ignore_index=True)
        
        # Handle duplicate reviewIds by keeping the most recent review
        unified_df = unified_df.sort_values('scrapedAt', ascending=False)
        unified_df = unified_df.drop_duplicates(subset=['reviewId'], keep='first')
        
        # Get translation configuration
        config = load_config()['google_maps']
        if config['api_settings'].get('translate', False):
            # Translate owner responses
            print(f"\nTranslating owner responses...")
            unified_df, characters_translated = translate_owner_responses(unified_df)
            print(f"Translated {characters_translated} characters in this run")
        
        # Save the unified reviews
        timestamp = datetime.now().strftime("%Y-%m-%d")
        output_file = f"reviews/google/allGoogleReviews_{timestamp}.xlsx"
        unified_df.to_excel(output_file, index=False)
        print(f"\nUnified reviews saved to {output_file}")
        print(f"Total reviews in unified file: {len(unified_df)}")
    else:
        print("No reviews found to unify.")

def update_establishment_base():
    # Read the latest unified reviews file
    google_dir = Path("reviews/google")
    latest_unified = max(google_dir.glob("allGoogleReviews_*.xlsx"))
    reviews_df = pd.read_excel(latest_unified)
    
    # Convert scrapedAt from ISO format to date
    reviews_df['scrapedAt'] = pd.to_datetime(reviews_df['scrapedAt'], format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')
    reviews_df['scrapedAt'] = reviews_df['scrapedAt'].dt.date
    
    # Drop any rows where scrapedAt couldn't be converted
    reviews_df = reviews_df.dropna(subset=['scrapedAt'])
    
    # Get the latest scraped date for each place
    latest_dates = reviews_df.groupby('placeId')['scrapedAt'].max().reset_index()
    
    # Read and update the establishment base
    base_df = pd.read_excel('establishments/establishment_base.xlsx')
    base_df['googleMapsScrapedAt'] = pd.to_datetime(base_df['googleMapsScrapedAt'], errors='coerce')
    base_df['googleMapsScrapedAt'] = base_df['googleMapsScrapedAt'].dt.date
    
    for _, row in latest_dates.iterrows():
        mask = base_df['placeId'] == row['placeId']
        if not base_df.loc[mask, 'googleMapsScrapedAt'].empty:
            current_date = base_df.loc[mask, 'googleMapsScrapedAt'].iloc[0]
            new_date = row['scrapedAt']
            if pd.isna(current_date) or new_date > current_date:
                base_df.loc[mask, 'googleMapsScrapedAt'] = new_date
    
    # Save the updated base
    base_df.to_excel('establishments/establishment_base.xlsx', index=False)

def main():
    config = load_config()['google_maps']
    establishments = get_establishments_to_scrape()
    if establishments.empty:
        print("No establishments to scrape based on the criteria.")
        return
    
    # Display selected establishments
    print("\nSelected establishments for scraping:")
    print("-" * 80)
    print(f"{'Title':<50} {'Place ID':<20} {'Review Count':<15}")
    print("-" * 80)
    
    total_reviews = 0
    for _, establishment in establishments.iterrows():
        print(f"{establishment['title']:<50} {establishment['placeId']:<20} {establishment['reviewsCount']:<15}")
        total_reviews += establishment['reviewsCount']
    
    print("-" * 80)
    print(f"Total establishments to scrape: {len(establishments)}")
    print(f"Total reviews to scrape: {total_reviews}")
    print("-" * 80)
    
    # Check if confirmation is required
    if config['scraping_criteria'].get('require_confirmation', True):
        while True:
            response = input("\nDo you want to proceed with scraping these establishments? (yes/no): ").lower()
            if response in ['yes', 'y']:
                print("\nStarting scraping process...")
                break
            elif response in ['no', 'n']:
                print("\nScraping cancelled by user.")
                return
            else:
                print("Please enter 'yes' or 'no'.")
    else:
        print("\nStarting scraping process automatically (require_confirmation is set to false)...")
    
    # Proceed with scraping
    reviews_df = scrape_reviews(establishments)
    save_reviews(reviews_df)
    unify_reviews()
    update_establishment_base()

if __name__ == "__main__":
    main() 