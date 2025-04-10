# Enhanced Review Analyzer

This project provides advanced review analysis for hair transplant clinics, utilizing data from both Google Maps and Trustpilot reviews to generate comprehensive scores and rankings.

## Key Features

- **Weighted Review Analysis**: Takes into account recency, reviewer credibility (local guides), and authenticity signals
- **Multiple Scoring Dimensions**: Analyzes ratings, volume, service quality, affordability, complaint handling and recommendations
- **Statistical Authenticity Detection**: Uses IQR-based analysis to identify suspicious review patterns
- **Context-Aware Complaint Detection**: Intelligently identifies complaints while handling negations
- **Data Visualization**: Creates insightful visualizations of the analysis results
- **Comprehensive Reports**: Generates detailed Excel reports with all metrics

## Metrics and Scoring System

The analyzer calculates the following key metrics:

### 1. Rating Score
- Weighted average rating that prioritizes recent reviews and trusted reviewers
- Penalizes suspicious review patterns
- Normalized using min-max scaling

### 2. Volume Score
- Measures the quantity of reviews with appropriate weighting
- Higher volume leads to higher confidence in the establishment's metrics

### 3. Complaint Score
- Tracks negative feedback and how well the establishment responds
- Formula: `% complaints - % only has response × 0.5 - % resolved × 0.75`
- Higher scores indicate better complaint handling

### 4. Service Score
- Measures the balance of positive vs. negative sentiment in reviews
- Formula: `(positive_reviews - negative_reviews) / total_reviews`

### 5. Affordability Score
- Evaluates price-related sentiments in reviews
- Formula: `(positive_price_mentions - negative_price_mentions) / total_price_mentions`

### 6. Recommendation Score
- Measures explicit recommendations or warnings in reviews
- Formula: `(recommendations - warnings) / total_recommendation_mentions`

### Final Composite Score
The final score combines all individual metrics with the following weighting:

```
final_score = 
    (rating_score × 0.5) 
  + volume_score × 0.15 
  + (complaint_score × 0.25 
     + service_score × 0.15 
     + affordability_score × 0.15 
     + recommendation_score × 0.15) × 0.35
```

## Advanced Features

### Context-Aware Complaint Detection

The system employs an advanced context-aware approach to identify complaints rather than just looking for keywords:

1. **Sentence-Level Analysis**: Reviews are split into sentences to analyze local context
2. **Negation Detection**: System detects negations like "no problems" or "didn't have any issues"
3. **Sentiment Cross-Verification**: A sentence with complaint keywords but positive sentiment is not counted as a complaint
4. **Multiple Criteria**: A true complaint must meet several conditions (complaint keyword + no negation + negative sentiment)

### Advanced Statistical Authenticity Detection

The system uses sophisticated statistical techniques to identify potential review manipulation:

1. **Clustering Analysis**: Uses DBSCAN density-based clustering to identify groups of similar reviews
2. **IQR-Based Outlier Detection**: Uses interquartile range analysis to identify:
   - Suspiciously large clusters (above the upper IQR bound for cluster sizes)
   - Suspiciously similar reviews (above the upper IQR bound for similarity scores)
3. **Two-Level Penalties**:
   - Reviews in suspicious clusters receive a weight penalty of 0.85
   - Individually suspicious reviews receive a stronger weight penalty of 0.5
   - Individual suspicion overrides cluster suspicion

This statistical approach ensures that authenticity detection is both sensitive and specific, with penalties proportional to the level of suspicion.

## Weighting Factors

The analysis applies the following weights to improve accuracy:

### Recency Weights
- Last month: 1.0 (full weight)
- Last quarter: 0.95
- Last 6 months: 0.9
- Last year: 0.85
- Last 2 years: 0.8
- Last 3 years: 0.7
- Older: 0.5

### Reviewer Credibility
- Local guides: 1.2× weight
- Regular users: 1.0× weight

### Authenticity
- Individually suspicious reviews: 0.5× weight
- Reviews in suspicious clusters: 0.85× weight
- Normal reviews: 1.0× weight

## Usage

### Setup

1. Ensure you have Python 3.8+ installed
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Download the required NLTK and spaCy resources:
   ```python
   import nltk
   nltk.download('vader_lexicon')
   nltk.download('punkt')
   nltk.download('stopwords')
   nltk.download('wordnet')
   import spacy
   spacy.cli.download('en_core_web_sm')
   ```

### Running the Analyzer

```python
from enhanced_review_analyzer import EnhancedReviewAnalyzer

# Initialize with data sources
analyzer = EnhancedReviewAnalyzer(
    google_maps_file="path/to/google_reviews.xlsx",
    trustpilot_file="path/to/trustpilot_reviews.xlsx",
    establishment_file="path/to/establishments.xlsx"
)

# Run the full analysis
results = analyzer.run_analysis()

# Save results to output directory
analyzer.save_results("output_directory")

# Access specific metrics
rating_metrics = analyzer.rating_metrics_df
volume_metrics = analyzer.volume_metrics_df
complaint_metrics = analyzer.complaint_metrics_df
service_metrics = analyzer.service_metrics_df
affordability_metrics = analyzer.affordability_metrics_df
recommendation_metrics = analyzer.recommendation_metrics_df

# Get the top 10 establishments
top_10 = results.head(10)
print(top_10[['rank', 'establishment_name', 'final_score']])
```

### Analyzing Authenticity

```python
# Get detailed authenticity metrics
authenticity_data = analyzer.establishment_authenticity

# View establishments with suspicious patterns
high_risk = authenticity_data[authenticity_data['authenticity_risk'].isin(['High', 'Very High'])]
print(high_risk[['establishment_name', 'suspicious_review_ratio', 'authenticity_risk']])

# Visualize authenticity patterns
analyzer.visualize_authenticity_clusters("output_directory")
```

## Output Files

The analyzer generates the following output files:

1. `establishment_rankings_TIMESTAMP.xlsx`: Main rankings with final scores
2. `establishment_detailed_metrics_TIMESTAMP.xlsx`: Detailed metrics for all calculations
3. Visualization files (if enabled):
   - `final_score_chart.png`: Bar chart of top establishments by final score
   - `component_scores_heatmap.png`: Heatmap of different scoring components
   - `volume_vs_rating.png`: Scatter plot of review volume vs. rating
   - `authenticity_bubble_chart.png`: Visualization of suspicious review patterns
   - `authenticity_risk_distribution.png`: Distribution of authenticity risk levels
   - `authenticity_risk_heatmap.png`: Heatmap of authenticity risk factors

## Example

A test script is provided to demonstrate the usage:

```bash
python test_enhanced_analyzer.py
```

This will:
1. Load and analyze the review data
2. Calculate all metrics and scores
3. Generate visualizations
4. Save results to the `analysis_results` directory

## License

This project is licensed under the MIT License - see the LICENSE file for details.
