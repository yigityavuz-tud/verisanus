# Hair Transplant Review Analysis System

A comprehensive system for collecting, translating, analyzing, and ranking hair transplant clinics based on online reviews from Google Maps and Trustpilot.

## System Overview

This project provides an end-to-end solution for:
1. **Data Collection**: Automated scraping of reviews from Google Maps and Trustpilot
2. **Text Processing**: Translation of non-English reviews for consistent analysis
3. **Review Analysis**: Advanced NLP analysis of review content with weighted metrics
4. **Authenticity Detection**: Statistical detection of suspicious review patterns
5. **Establishment Ranking**: Comprehensive scoring and ranking of hair transplant clinics

## Installation

### Requirements

1. Python 3.8+ is required
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Download NLP resources:
   ```python
   import nltk
   nltk.download('vader_lexicon')
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   
   import spacy
   spacy.cli.download('en_core_web_sm')
   ```

4. Set up API tokens:
   - Create a `tokens` directory
   - Add required API token files:
     - `apify_token_dev2.txt` (for web scraping)
     - `deepl_token_dev.txt` (for translation)
     - `deepseek_api_key.txt` (alternative translation)

## System Components

### 1. Data Collection

#### Google Maps Reviews (`scrape_google_reviews.py`)

Collects reviews from Google Maps for specified hair transplant establishments.

Features:
- Configurable scraping based on criteria in `config.yaml`
- Targeted scraping by place ID or by review count thresholds
- Support for incremental scraping with date-based cutoffs
- Automatic deduplication of reviews
- Handling of review responses

Usage:
```python
import scrape_google_reviews as sgr

# Get establishments to scrape based on config criteria
establishments = sgr.get_establishments_to_scrape()

# Scrape reviews
reviews_df = sgr.scrape_reviews(establishments)

# Save individual review file
sgr.save_reviews(reviews_df)

# Unify with previously collected reviews
sgr.unify_reviews()

# Update establishment database with new scrape dates
sgr.update_establishment_base()
```

#### Trustpilot Reviews (`scrape_trustpilot_reviews.py`)

Collects reviews from Trustpilot for specified establishments.

Features:
- Website-based matching for establishments
- Configurable scraping parameters
- Support for review responses and ratings
- Review ID generation for deduplication

Usage:
```python
import scrape_trustpilot_reviews as str

# Get establishments with Trustpilot presence
establishments = str.get_establishments_to_scrape()

# Scrape reviews
trustpilot_reviews = str.scrape_reviews(establishments)

# Save individual review file
str.save_reviews(trustpilot_reviews)

# Unify with previously collected reviews
str.unify_reviews()

# Update establishment database
str.update_establishment_base()
```

### 2. Text Processing

#### Translation Utilities (`translation_utils.py`)

Handles translation of non-English reviews to ensure consistent analysis.

Features:
- Support for multiple translation services (DeepL, DeepSeek)
- Batch processing of reviews
- Character counting for API usage tracking
- Configurable translation targets

Usage:
```python
from translation_utils import translate_column

# Translate a column in a DataFrame
translated_df, characters_translated = translate_column(
    input_data=df,
    source_column="reviewBody",
    target_column="reviewBody_en",
    translation_service="deepseek"
)
```

### 3. Review Analysis

#### Basic Review Analyzer (`review_analyzer.py`)

The foundation review analysis system with core metrics.

Features:
- Combined analysis of Google Maps and Trustpilot reviews
- Sentiment analysis of review text
- Topic modeling and word choice analysis
- Temporal pattern analysis
- Basic authenticity checks

#### Enhanced Review Analyzer (`enhanced_review_analyzer.py`)

Advanced version with improved scoring and authenticity detection.

Features:
- **Context-Aware Complaint Detection**: Intelligently identifies complaints handling negations
- **IQR-Based Authenticity Detection**: Statistical identification of suspicious reviews
- **Advanced Scoring System**: Weighted metrics accounting for recency, credibility, and authenticity
- **Specialized Metrics**: Service, affordability, and recommendation scores
- **Visualization Capabilities**: Generated graphs for analysis insights

Usage:
```python
from enhanced_review_analyzer import EnhancedReviewAnalyzer

# Initialize analyzer with data sources
analyzer = EnhancedReviewAnalyzer(
    google_maps_file="reviews/google/allGoogleReviews_2025-04-06.xlsx",
    trustpilot_file="reviews/trustpilot/allTrustpilotReviews_2025-04-06.xlsx",
    establishment_file="establishments/establishment_base.xlsx"
)

# Run full analysis
results = analyzer.run_analysis()

# Save results
analyzer.save_results("analysis_results")
```

## Scoring System

The enhanced analyzer calculates the following key metrics:

### 1. Rating Score
- Weighted average rating that accounts for:
  - Review recency (newer reviews weighted higher)
  - Reviewer credibility (local guides given 1.2× weight)
  - Review authenticity (suspicious reviews penalized)

### 2. Volume Score
- Weighted measure of review quantity
- Adjusts for recency, credibility, and authenticity

### 3. Complaint Score
- Formula: `% complaints - % only has response × 0.5 - % resolved × 0.75`
- Uses context-aware complaint detection to handle negations
- Recognizes complaint resolution in responses

### 4. Service Score
- Measures positive vs. negative sentiment balance
- Formula: `(positive_reviews - negative_reviews) / total_reviews`

### 5. Affordability Score
- Evaluates price mentions in reviews
- Formula: `(positive_price_mentions - negative_price_mentions) / total_price_mentions`

### 6. Recommendation Score
- Tracks explicit recommendations in reviews
- Formula: `(recommendations - warnings) / total_recommendation_mentions`

### Final Composite Score

The final score combines all metrics with weighted importance:

```
final_score = 
    (rating_score × 0.5) 
  + volume_score × 0.15 
  + (complaint_score × 0.25 
     + service_score × 0.15 
     + affordability_score × 0.15 
     + recommendation_score × 0.15) × 0.35
```

## Configuration

The system is configured through `config.yaml` with sections for:

### Google Maps Scraping
```yaml
google_maps:
  scraping_criteria:
    scrape_unscraped: true
    scrape_before_date: "2024-01-01"
    max_reviews_per_establishment: 1500
    min_reviews_per_establishment: 100
    max_total_reviews: 9500
    require_confirmation: true
    custom_place_ids: []
  
  api_settings:
    translate: false
    apify_token_file: "tokens/apify_token_dev2.txt"
    # ... other settings
```

### Trustpilot Scraping
```yaml
trustpilot:
  scraping_criteria:
    scrape_unscraped: false
    scrape_before_date: "2024-04-07"
    max_establishments: 500
    custom_place_ids: []
  
  api_settings:
    # ... settings
```

### Analysis Weights
```yaml
analysis:
  composite_score_weights:
    # Rating and volume (30%)
    avg_rating: 17.5
    total_reviews: 7.5
    # ... other weights
```

## Data Flow

1. **Scraping Stage**:
   - Google Maps and Trustpilot reviews are collected using Apify API
   - Reviews are saved as Excel files with timestamps
   - Individual scrapes are unified into consolidated files

2. **Translation Stage**:
   - Non-English reviews are detected and translated
   - Translations are stored in dedicated columns (*_en)

3. **Analysis Stage**:
   - Reviews are filtered to focus on hair transplantation
   - Suspicious review patterns are detected using clustering
   - Multiple metrics are calculated with appropriate weighting
   - Establishments are scored and ranked

4. **Output Stage**:
   - Rankings and detailed metrics are saved as Excel files
   - Visualizations are generated to highlight key insights

## Testing

The `test_enhanced_analyzer.py` script demonstrates the complete analysis workflow:

```bash
python test_enhanced_analyzer.py
```

This will:
1. Load example data files
2. Run the enhanced analysis
3. Generate and display results
4. Create visualizations
5. Save all output to the `analysis_results` directory

## Troubleshooting

- **API Rate Limits**: The system includes character logging for translation services - monitor these logs
- **Memory Usage**: Running with large datasets may require increased memory for clustering operations
- **Missing Data**: Ensure all required files exist before running analysis

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Acknowledgments

- This system uses the Apify platform for web scraping
- Translation services provided by DeepL and DeepSeek
- NLP capabilities powered by NLTK, spaCy, and scikit-learn
