import os
import yaml
import pandas as pd
import hashlib
from datetime import datetime
from apify_client import ApifyClient
from pathlib import Path
# Import from translation_utils instead of direct deepl import
from translation_utils import translate_column, log_deepl_usage, load_config

def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

def get_apify_token(token_file):
    try:
        with open(token_file, 'r') as file:
            return file.read().strip()
    except FileNotFoundError:
        raise ValueError(f"Token file {token_file} not found. Please check the config file.")

def generate_review_id(company_domain, published_at):
    return hashlib.md5(f"{company_domain}_{published_at}".encode()).hexdigest()

def get_establishments_to_scrape():
    config = load_config()['trustpilot']
    df = pd.read_excel('establishments/establishment_base.xlsx')
    
    # Exclude establishments without a website URL
    df = df[df['website'].notna() & (df['website'] != '')]
    
    # Only include establishments where hasTrustpilot is 1
    df = df[df['hasTrustpilot'] == 1]
    
    # Convert trustpilotScrapedAt to datetime, handling empty values
    df['trustpilotScrapedAt'] = pd.to_datetime(df['trustpilotScrapedAt'], errors='coerce')
    
    # Handle custom placeIds if provided
    if config['scraping_criteria']['custom_place_ids']:
        df = df[df['placeId'].isin(config['scraping_criteria']['custom_place_ids'])]
    else:
        # Apply filtering criteria
        if config['scraping_criteria']['scrape_unscraped']:
            # If we're scraping unscraped, we don't need to check the date
            df = df[df['trustpilotScrapedAt'].isna() | (df['trustpilotScrapedAt'] == '') | (df['trustpilotScrapedAt'] < config['scraping_criteria']['scrape_before_date'])]

    # Limit the number of establishments
    df = df.head(config['scraping_criteria']['max_establishments'])
    
    return df

def prepare_trustpilot_url(website):
    # Remove https:// and add ?languages=all
    domain = website.replace('https://www.', '')
    return f"{domain}?languages=all"

def standardize_date(date_str):
    if pd.isna(date_str):
        return None
    try:
        # Try parsing with timezone info first
        date = pd.to_datetime(date_str, format='%Y-%m-%dT%H:%M:%S.%fZ', errors='coerce')
        if pd.isna(date):
            # Try parsing without timezone info
            date = pd.to_datetime(date_str, format='%Y-%m-%dT%H:%M:%S', errors='coerce')
        if pd.isna(date):
            # Try parsing with just date
            date = pd.to_datetime(date_str, format='%Y-%m-%d', errors='coerce')
        return date.date() if not pd.isna(date) else None
    except:
        return None

def scrape_reviews(establishments):
    config = load_config()['trustpilot']
    token = get_apify_token(config['api_settings']['apify_token_file'])
    client = ApifyClient(token)
    
    all_reviews = []
    for _, establishment in establishments.iterrows():
        company_domain = prepare_trustpilot_url(establishment['website'])
        
        run_input = {
            "companyDomain": company_domain,
            "startPage": config['api_settings']['start_page'],
            "count": config['api_settings']['count'],
            "minDelay": config['api_settings']['minDelay'],
            "replies": config['api_settings']['replies'],
            "sort": config['api_settings']['sort'],
            "verified": config['api_settings']['verified']
        }
        
        run = client.actor("fLXimoyuhE1UQgDbM").call(run_input=run_input)
        
        for item in client.dataset(run["defaultDatasetId"]).iterate_items():
            # Standardize dates
            item['datePublished'] = standardize_date(item.get('datePublished'))
            item['experienceDate'] = standardize_date(item.get('experienceDate'))
            item['replyPublishedDate'] = standardize_date(item.get('replyPublishedDate'))
            item['replyUpdatedDate'] = standardize_date(item.get('replyUpdatedDate'))
            
            # Generate review ID using standardized date
            item['reviewId'] = generate_review_id(company_domain, item['datePublished'])
            item['placeId'] = establishment['placeId']  # Add placeId to link back to establishment
            item['trustpilotScrapedAt'] = datetime.now().date()  # Add scraping timestamp
            
            # Keep only the fields we want
            review_data = {
                'reviewId': item['reviewId'],
                'placeId': item['placeId'],
                'consumerCountryCode': item.get('consumerCountryCode', ''),
                'datePublished': item['datePublished'],
                'experienceDate': item['experienceDate'],
                'likes': item.get('likes', 0),
                'numberOfReviews': item.get('numberOfReviews', 0),
                'ratingValue': item.get('ratingValue', 0),
                'replyMessage': item.get('replyMessage', ''),
                'replyPublishedDate': item['replyPublishedDate'],
                'replyUpdatedDate': item['replyUpdatedDate'],
                'reviewBody': item.get('reviewBody', ''),
                'reviewHeadline': item.get('reviewHeadline', ''),
                'reviewLanguage': item.get('reviewLanguage', ''),
                'trustpilotScrapedAt': item['trustpilotScrapedAt']
            }
            
            all_reviews.append(review_data)
    
    reviews_df = pd.DataFrame(all_reviews)
    return reviews_df

def save_reviews(reviews_df):
    # Define the columns we want to keep
    required_columns = [
        'reviewId',
        'placeId',
        'consumerCountryCode',
        'datePublished',
        'experienceDate',
        'likes',
        'numberOfReviews',
        'ratingValue',
        'replyMessage',
        'replyPublishedDate',
        'replyUpdatedDate',
        'reviewBody',
        'reviewHeadline',
        'reviewLanguage',
        'trustpilotScrapedAt'
    ]
    
    # Keep only the required columns
    reviews_df = reviews_df[required_columns]
    
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M")
    output_file = f"reviews/trustpilot/trustpilotReviews_{timestamp}.xlsx"
    
    # Ensure the directory exists
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Save the reviews
    reviews_df.to_excel(output_file, index=False)

def count_characters(text):
    """Count the number of characters in a text, handling NaN values."""
    if pd.isna(text):
        return 0
    return len(str(text))

def log_deepl_usage(characters_translated):
    """Log the total characters translated to a file."""
    log_file = Path("tokens/deepl_usage_log.txt")
    
    # Get current date
    current_date = datetime.now().strftime("%Y-%m-%d")
    
    # Read existing log if it exists
    if log_file.exists():
        with open(log_file, 'r') as f:
            lines = f.readlines()
            if lines:
                last_line = lines[-1].strip()
                try:
                    total_chars = int(last_line.split(': ')[1])
                except (IndexError, ValueError):
                    total_chars = 0
            else:
                total_chars = 0
    else:
        total_chars = 0
    
    # Add new characters to total
    total_chars += characters_translated
    
    # Append new log entry
    with open(log_file, 'a') as f:
        f.write(f"{current_date}: {total_chars}\n")
    
    return total_chars

def translate_texts(df):
    """
    Translate non-English texts to English using the configured translation service.
    Only translates rows where reviewLanguage is not 'en'.
    Returns the translated DataFrame and the number of characters translated.
    """
    config = load_config()['trustpilot']
    translation_service = config['api_settings'].get('translation_service', 'deepl')
    
    # Initialize English translations if they don't exist
    if 'reviewBody_en' not in df.columns:
        df['reviewBody_en'] = df['reviewBody']
    if 'reviewHeadline_en' not in df.columns:
        df['reviewHeadline_en'] = df['reviewHeadline']
    if 'replyMessage_en' not in df.columns:
        df['replyMessage_en'] = df['replyMessage']
    
    characters_translated = 0
    
    # Only translate non-English reviews
    to_translate = df[df['reviewLanguage'] != 'en']
    
    if not to_translate.empty:
        print(f"Translating {len(to_translate)} reviews using {translation_service}...")
        
        # Translate reviewBody
        if to_translate['reviewBody'].notna().any():
            body_df, body_chars = translate_column(
                to_translate,
                'reviewBody',
                'reviewBody_en',
                translation_service=translation_service
            )
            df.loc[to_translate.index, 'reviewBody_en'] = body_df['reviewBody_en']
            characters_translated += body_chars
        
        # Translate reviewHeadline
        if to_translate['reviewHeadline'].notna().any():
            headline_df, headline_chars = translate_column(
                to_translate,
                'reviewHeadline',
                'reviewHeadline_en',
                translation_service=translation_service
            )
            df.loc[to_translate.index, 'reviewHeadline_en'] = headline_df['reviewHeadline_en']
            characters_translated += headline_chars
        
        # Translate replyMessage
        if to_translate['replyMessage'].notna().any():
            reply_df, reply_chars = translate_column(
                to_translate,
                'replyMessage',
                'replyMessage_en',
                translation_service=translation_service
            )
            df.loc[to_translate.index, 'replyMessage_en'] = reply_df['replyMessage_en']
            characters_translated += reply_chars
    
    return df, characters_translated

def unify_reviews():
    trustpilot_dir = Path("reviews/trustpilot")
    all_reviews = []
    
    # Define the columns we want to keep, including the translated columns
    required_columns = [
        'reviewId',
        'placeId',
        'consumerCountryCode',
        'datePublished',
        'experienceDate',
        'likes',
        'numberOfReviews',
        'ratingValue',
        'replyMessage',
        'replyMessage_en',
        'replyPublishedDate',
        'replyUpdatedDate',
        'reviewBody',
        'reviewBody_en',
        'reviewHeadline',
        'reviewHeadline_en',
        'reviewLanguage',
        'trustpilotScrapedAt'
    ]
    
    # First, try to load the latest unified file if it exists
    latest_unified = list(trustpilot_dir.glob("allTrustpilotReviews_*.xlsx"))
    if latest_unified:
        latest_unified = max(latest_unified)
        existing_df = pd.read_excel(latest_unified)
        print(f"Loaded existing unified reviews from {latest_unified}")
        # Ensure all required columns exist in the existing DataFrame
        for col in required_columns:
            if col not in existing_df.columns:
                existing_df[col] = None
        all_reviews.append(existing_df)
    
    # Then load all individual review files
    for file in trustpilot_dir.glob("trustpilotReviews_*.xlsx"):
        if file.name.startswith("allTrustpilotReviews_"):
            continue  # Skip unified files
        df = pd.read_excel(file)
        # Ensure all required columns exist in the DataFrame
        for col in required_columns:
            if col not in df.columns:
                df[col] = None
        # Keep only the required columns
        df = df[required_columns]
        all_reviews.append(df)
    
    if all_reviews:
        unified_df = pd.concat(all_reviews, ignore_index=True)
        
        # Handle duplicate reviewIds by keeping the most recent review
        unified_df = unified_df.sort_values('trustpilotScrapedAt', ascending=False)
        unified_df = unified_df.drop_duplicates(subset=['reviewId'], keep='first')
        
        # Get translation configuration
        config = load_config()['trustpilot']
        if config['api_settings']['translate']:
            # Translate texts for non-English reviews
            unified_df, characters_translated = translate_texts(unified_df)
            
            print(f"Translated {characters_translated} characters in this run")
        
        # Ensure all required columns are present in the final DataFrame
        for col in required_columns:
            if col not in unified_df.columns:
                unified_df[col] = None
        
        # Save the unified reviews with all columns
        timestamp = datetime.now().strftime("%Y-%m-%d")
        unified_df.to_excel(f"reviews/trustpilot/allTrustpilotReviews_{timestamp}.xlsx", index=False)
        print(f"Unified reviews saved to reviews/trustpilot/allTrustpilotReviews_{timestamp}.xlsx")
        print(f"Total reviews: {len(unified_df)}")

def update_establishment_base():
    # Read the latest unified reviews file
    trustpilot_dir = Path("reviews/trustpilot")
    latest_unified = max(trustpilot_dir.glob("allTrustpilotReviews_*.xlsx"))
    reviews_df = pd.read_excel(latest_unified)
    
    # Get the latest scraped date for each place
    latest_dates = reviews_df.groupby('placeId')['datePublished'].max().reset_index()
    
    # Read and update the establishment base
    base_df = pd.read_excel('establishments/establishment_base.xlsx')

    # Convert dates to datetime.date objects for consistent comparison
    base_df['trustpilotScrapedAt'] = pd.to_datetime(base_df['trustpilotScrapedAt'], errors='coerce').dt.date
    latest_dates['datePublished'] = pd.to_datetime(latest_dates['datePublished'], errors='coerce').dt.date
    
    for _, row in latest_dates.iterrows():
        mask = base_df['placeId'] == row['placeId']
        if not base_df.loc[mask, 'trustpilotScrapedAt'].empty:
            current_date = base_df.loc[mask, 'trustpilotScrapedAt'].iloc[0]
            new_date = row['datePublished']
            if pd.isna(current_date) or (not pd.isna(new_date) and new_date > current_date):
                base_df.loc[mask, 'trustpilotScrapedAt'] = new_date

    # Count the number of reviews for each placeId
    trustpilotReviewCount = reviews_df.groupby('placeId').size().reset_index(name='trustpilotReviewCount')
    print(f"Number of reviews per establishment: {trustpilotReviewCount}")
    
    # Merge the review counts into the establishment base excluding the trustpilotReviewCount column
    base_df = base_df.drop(columns=['trustpilotReviewCount'], errors='ignore')
    base_df = base_df.merge(trustpilotReviewCount, on='placeId', how='left')
    
    # Fill NaN values in reviewCount with 0 (for establishments with no reviews)
    base_df['trustpilotReviewCount'] = base_df['trustpilotReviewCount'].fillna(0).astype(int)
    
    # Save the updated base
    base_df.to_excel('establishments/establishment_base.xlsx', index=False)

def main():
    establishments = get_establishments_to_scrape()
    if establishments.empty:
        print("No establishments to scrape based on the criteria.")
        return
    
    reviews_df = scrape_reviews(establishments)
    save_reviews(reviews_df)
    unify_reviews()
    update_establishment_base()

if __name__ == "__main__":
    main() 