"""
Enhanced version of the ReviewAnalyzer class with updated scoring metrics.
This extends the original review_analyzer.py with new scoring mechanisms.

Version: 1.3.0
Key Developments:
1. Enhanced Authenticity Detection:
   - Advanced clustering using DBSCAN for suspicious review detection
   - Better validation and error handling for clustering process

2. Communication Score:
   - Renamed to better reflect what it measures
   - Enhanced scoring logic for better communication assessment
   - Added weighted metrics for response quality
   - Improved handling of resolution language detection

3. Enhanced Composite Score Calculation:
   - Updated weighting system for better balance
   - Added communication score as a key component
   - Improved normalization of individual scores

4. Additional Features:
   - Enhanced review text preprocessing
   - Improved time-based weighting system

5. Code Improvements:
   - Better error handling throughout
   - More robust input validation
   - Improved documentation

TODO:
- Add config file support for weights and parameters
"""

import pandas as pd
import numpy as np
from datetime import datetime
import re
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk import ngrams
import spacy
import warnings
import math
from scipy.stats import entropy
from textblob import TextBlob
import openpyxl
import yaml
import os

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Load spaCy model for NER and language processing
nlp = spacy.load('en_core_web_sm')

# Suppress warnings
warnings.filterwarnings('ignore')

class EnhancedReviewAnalyzer:
    def __init__(self, google_maps_file, trustpilot_file, establishment_file, config_file='config.yaml'):
        """
        Initialize the EnhancedReviewAnalyzer with the paths to the data files.
        
        Parameters:
        -----------
        google_maps_file : str
            Path to the Google Maps reviews CSV file
        trustpilot_file : str
            Path to the Trustpilot reviews CSV file
        establishment_file : str
            Path to the establishment base data CSV file
        config_file : str
            Path to the configuration file
        """
        self.google_maps_file = google_maps_file
        self.trustpilot_file = trustpilot_file
        self.establishment_file = establishment_file
        
        # Load configuration
        with open(config_file, 'r') as f:
            self.config = yaml.safe_load(f)
        
        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Define business aspects for aspect-based sentiment analysis
        self.aspects = {
            'service': ['service', 'staff', 'employee', 'waiter', 'waitress', 'server', 'customer service'],
            'quality': ['quality', 'excellent', 'great', 'good', 'bad', 'poor', 'terrible'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'money'],
            'ambiance': ['atmosphere', 'environment', 'decor', 'music', 'noise', 'quiet'],
            'cleanliness': ['clean', 'dirty', 'filthy', 'hygiene', 'sanitary', 'neat', 'tidy'],
            'location': ['location', 'area', 'neighborhood', 'parking', 'accessible', 'central']
        }
        
        # Define time weights for recency (updated as per requirements)
        self.time_weights = {
            30: 1.0,    # Last month: full weight
            90: 0.95,   # Last quarter: 95%
            180: 0.9,   # Last 6 months: 90%
            365: 0.85,  # Last year: 85%
            730: 0.8,   # Last 2 years: 80%
            1095: 0.7,  # Last 3 years: 70%
            float('inf'): 0.5  # Older: 50%
        }
        
        # Define recommendation keywords
        self.recommendation_keywords = {
            'positive': ['recommend', 'recommended', 'worth', 'must visit', 'must try', 
                         'should visit', 'should try', 'great choice', 'good choice'],
            'negative': ['avoid', 'stay away', 'waste', 'don\'t recommend', 'not recommend', 
                         'wouldn\'t recommend', 'not worth', 'skip']
        }
        
        # Define price evaluation keywords
        self.price_keywords = {
            'positive': ['worth', 'fair price', 'reasonable', 'affordable', 'good value', 
                         'great value', 'value for money', 'inexpensive', 'economical', 'cheap'],
            'negative': ['expensive', 'overpriced', 'pricey', 'not worth', 'overcharged', 
                         'costly', 'high price', 'steep price', 'too much']
        }
        
        # Define complaint keywords
        self.complaint_keywords = [
            'problem', 'issue', 'complaint', 'disappointing', 'disappointed',
            'frustrated', 'terrible', 'awful', 'bad', 'wrong', 'mistake',
            'error', 'fail', 'poor', 'unacceptable', 'dissatisfied'
        ]
        
        # Define resolution keywords
        self.resolution_keywords = [
            'resolve', 'resolved', 'solution', 'solved', 'fix', 'fixed',
            'addressed', 'corrected', 'improved', 'apologize', 'sorry',
            'refund', 'replacement', 'compensate', 'compensation'
        ]
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
        # Prepare special metrics for scoring
        self.prepare_review_labels()
        
    def load_data(self):
        """Load the data files into pandas DataFrames."""
        print("Loading data...")
        
        # Load Google Maps reviews
        self.google_df = pd.read_excel(self.google_maps_file)
        
        # Load Trustpilot reviews
        self.trustpilot_df = pd.read_excel(self.trustpilot_file)
        
        # Load establishment data
        self.establishments_df = pd.read_excel(self.establishment_file)
        
        print(f"Loaded {len(self.google_df)} Google Maps reviews")
        print(f"Loaded {len(self.trustpilot_df)} Trustpilot reviews")
        print(f"Loaded {len(self.establishments_df)} establishments")
        
    def preprocess_data(self):
        """Preprocess the data for analysis."""
        print("Preprocessing data...")
        
        # Convert date columns to datetime
        self.google_df['publishedAtDate'] = pd.to_datetime(self.google_df['publishedAtDate'], errors='coerce')
        self.google_df['responseFromOwnerDate'] = pd.to_datetime(self.google_df['responseFromOwnerDate'], errors='coerce')
        
        self.trustpilot_df['datePublished'] = pd.to_datetime(self.trustpilot_df['datePublished'], errors='coerce')
        self.trustpilot_df['replyPublishedDate'] = pd.to_datetime(self.trustpilot_df['replyPublishedDate'], errors='coerce')
        
        # Use English translated text where available
        self.google_df['review_text'] = self.google_df['textTranslated'].fillna(self.google_df['text'])
        self.trustpilot_df['review_text'] = self.trustpilot_df['reviewBody_en'].fillna(self.trustpilot_df['reviewBody'])
        
        # Fill missing values
        self.google_df['responseFromOwnerText'] = self.google_df['responseFromOwnerText'].fillna('')
        self.trustpilot_df['replyMessage'] = self.trustpilot_df['replyMessage'].fillna('')
        
        # Create combined review dataset
        self.process_combined_reviews()

        # Filter out reviews unrelated to hair transplantation
        self.filter_hair_transplant_reviews()
        
        # Run authenticity detection to identify suspicious reviews
        self.detect_review_authenticity()
        
        print("Data preprocessing completed")
        
    def process_combined_reviews(self):
        """Create a combined review dataset with normalized fields."""
        # Extract relevant fields from Google Maps
        google_reviews = self.google_df[['placeId', 'stars', 'publishedAtDate', 
                                       'reviewerNumberOfReviews', 'isLocalGuide',
                                       'review_text', 'responseFromOwnerText', 
                                       'responseFromOwnerDate']].copy()
        
        google_reviews['source'] = 'Google Maps'
        google_reviews['rating'] = google_reviews['stars']
        google_reviews['review_date'] = pd.to_datetime(google_reviews['publishedAtDate']).dt.tz_localize(None)  # Make timezone-naive
        google_reviews['response_text'] = google_reviews['responseFromOwnerText']
        google_reviews['response_date'] = pd.to_datetime(google_reviews['responseFromOwnerDate']).dt.tz_localize(None)  # Make timezone-naive
        google_reviews['reviewer_experience'] = google_reviews['reviewerNumberOfReviews']
        # Convert isLocalGuide to boolean
        google_reviews['isLocalGuide'] = google_reviews['isLocalGuide'].fillna(False).astype(bool)
        
        # Extract relevant fields from Trustpilot
        trustpilot_reviews = self.trustpilot_df[['placeId', 'ratingValue', 'datePublished', 
                                              'numberOfReviews', 'review_text',
                                              'replyMessage', 'replyPublishedDate']].copy()
        
        trustpilot_reviews['source'] = 'Trustpilot'
        trustpilot_reviews['rating'] = trustpilot_reviews['ratingValue']
        trustpilot_reviews['review_date'] = pd.to_datetime(trustpilot_reviews['datePublished']).dt.tz_localize(None)  # Make timezone-naive
        trustpilot_reviews['response_text'] = trustpilot_reviews['replyMessage']
        trustpilot_reviews['response_date'] = pd.to_datetime(trustpilot_reviews['replyPublishedDate']).dt.tz_localize(None)  # Make timezone-naive
        trustpilot_reviews['reviewer_experience'] = trustpilot_reviews['numberOfReviews']
        trustpilot_reviews['isLocalGuide'] = False  # Trustpilot doesn't have local guides
        
        # Combine the reviews
        combined_cols = ['placeId', 'source', 'rating', 'review_date', 'review_text', 
                        'response_text', 'response_date', 'reviewer_experience', 
                        'isLocalGuide']
        
        self.google_reviews = google_reviews[combined_cols]
        self.trustpilot_reviews = trustpilot_reviews[combined_cols]
        self.combined_reviews = pd.concat([self.google_reviews, self.trustpilot_reviews], ignore_index=True)
        
        # Calculate days since review using timezone-naive datetime
        current_date = datetime.now()
        self.combined_reviews['days_since_review'] = (current_date - self.combined_reviews['review_date']).dt.days
        
        # Calculate response time in days
        has_response = ~self.combined_reviews['response_date'].isna()
        self.combined_reviews['response_time_days'] = np.nan
        self.combined_reviews.loc[has_response, 'response_time_days'] = (
            self.combined_reviews.loc[has_response, 'response_date'] - 
            self.combined_reviews.loc[has_response, 'review_date']
        ).dt.days
        
        # Filter out invalid response times (negative or extremely large)
        invalid_response_time = (self.combined_reviews['response_time_days'] < 0) | (self.combined_reviews['response_time_days'] > 365)
        self.combined_reviews.loc[invalid_response_time, 'response_time_days'] = np.nan
        
        # Compute review text length
        self.combined_reviews['review_length'] = self.combined_reviews['review_text'].fillna('').str.len()
        self.combined_reviews['response_length'] = self.combined_reviews['response_text'].fillna('').str.len()
        
        # Calculate time weight based on recency
        self.combined_reviews['time_weight'] = self.combined_reviews['days_since_review'].apply(self.get_time_weight)
        
        # Initialize suspicion flag (will be updated after authenticity detection)
        self.combined_reviews['is_suspicious'] = False
        
        # Initialize local guide weight
        self.combined_reviews['local_guide_weight'] = self.combined_reviews['isLocalGuide'].apply(lambda x: 1.2 if x else 1.0)
        
        print(f"Combined dataset created with {len(self.combined_reviews)} reviews")
    
    def get_time_weight(self, days_since_review):
        """Calculate time weight based on days since review."""
        if pd.isna(days_since_review):
            return 0.5  # Default weight for unknown dates
            
        for max_days, weight in sorted(self.time_weights.items()):
            if days_since_review <= max_days:
                return weight
                
        return self.time_weights[float('inf')]
    
    def filter_hair_transplant_reviews(self):
        """
        Identifies and keeps only reviews related to hair transplantation services.
        Removes reviews about other medical procedures like dental work or plastic surgery
        that don't mention hair.
        """
        print("Filtering reviews to keep only hair transplantation related content...")
        
        # Keywords related to hair transplantation
        hair_keywords = [
            'hair', 'transplant', 'graft', 'follicle', 'fue', 'dhi', 'balding', 
            'hairline', 'donor', 'recipient', 'scalp', 'baldness', 'crown', 
            'receding', 'density', 'implant', 'alopecia', 'shedding'
        ]
        
        # Keywords indicating unrelated medical procedures
        other_procedure_keywords = [
            'teeth', 'dental', 'veneers', 'implants', 'crown', 'filling',
            'plastic surgery', 'rhinoplasty', 'nose job', 'facelift', 'botox',
            'liposuction', 'tummy tuck', 'breast', 'augmentation', 'laser eye'
        ]
        
        # Count before filtering
        initial_count = len(self.combined_reviews)
        
        # Create a mask for reviews to keep
        keep_reviews = []
        
        for idx, review in self.combined_reviews.iterrows():
            if pd.isna(review['review_text']) or review['review_text'] == '':
                # Keep empty reviews (will be handled elsewhere)
                keep_reviews.append(True)
                continue
                
            text = review['review_text'].lower()
            
            # Check if review mentions hair transplantation
            is_hair_related = any(keyword in text for keyword in hair_keywords)
            
            # Check if review mentions other medical procedures but NOT hair
            is_other_procedure = (not is_hair_related) and any(keyword in text for keyword in other_procedure_keywords)
            
            # Keep reviews that are hair-related OR not about other medical procedures
            keep_reviews.append(is_hair_related or not is_other_procedure)
        
        # Apply filter
        self.combined_reviews = self.combined_reviews[keep_reviews]
        
        # Count after filtering
        filtered_count = initial_count - len(self.combined_reviews)
        
        print(f"Filtered out {filtered_count} reviews unrelated to hair transplantation")
        print(f"Remaining reviews: {len(self.combined_reviews)}")

    def detect_review_authenticity(self):
        """
        Enhanced authenticity detection using clustering to identify suspicious reviews.
        Uses IQR-based outlier detection for both cluster sizes and similarity scores.
        This method populates the 'is_suspicious' flag and 'suspicion_weight' in combined_reviews.
        """
        print("Detecting suspicious reviews using advanced clustering...")
        
        from sklearn.cluster import DBSCAN
        from sklearn.metrics.pairwise import cosine_similarity
        
        # Initialize tracking for all cluster data
        all_clusters_data = []
        suspicious_clusters = []
        suspicious_reviews_list = []
        
        # Add input validation
        if len(self.combined_reviews) == 0:
            print("No reviews available for authenticity analysis")
            return
        
        # Process each establishment separately
        for place_id in self.combined_reviews['placeId'].unique():
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            if len(place_reviews) < 5:
                continue
            
            # Get non-empty review texts
            valid_reviews = place_reviews[place_reviews['review_text'].notna() & 
                                        (place_reviews['review_text'].str.strip() != '')]
            valid_indices = valid_reviews.index
            
            if len(valid_reviews) < 5:
                continue
            
            try:
                # Create TF-IDF vectorizer
                vectorizer = TfidfVectorizer(
                    max_df=0.9,
                    min_df=2,
                    stop_words='english',
                    max_features=5000,
                    ngram_range=(1, 2)
                )
                
                # Create TF-IDF matrix
                tfidf_matrix = vectorizer.fit_transform(valid_reviews['review_text'])
                
                # Add validation for empty matrix
                if tfidf_matrix.shape[0] == 0:
                    continue
                
                # Calculate cosine similarity matrix
                similarity_matrix = cosine_similarity(tfidf_matrix)
                
                # Debug: Check for negative values in similarity matrix
                if np.any(similarity_matrix < 0):
                    print(f"Warning: Negative values found in similarity matrix for place_id {place_id}")
                    print(f"Min similarity: {np.min(similarity_matrix)}, Max similarity: {np.max(similarity_matrix)}")
                    # Clip negative values to 0
                    similarity_matrix = np.clip(similarity_matrix, 0, 1)
                
                # Convert similarity to distance (1 - similarity)
                # Ensure no negative values in distance matrix
                distance_matrix = 1 - similarity_matrix
                distance_matrix = np.clip(distance_matrix, 0, 1)  # Ensure all values are between 0 and 1
                
                # Debug: Check distance matrix
                if np.any(distance_matrix < 0):
                    print(f"Warning: Negative values found in distance matrix for place_id {place_id}")
                    print(f"Min distance: {np.min(distance_matrix)}, Max distance: {np.max(distance_matrix)}")
                    continue
                
                # Apply DBSCAN clustering with adjusted parameters
                # eps=0.3 means reviews with distance less than 0.3 (similarity > 0.7) are considered neighbors
                # min_samples=2 means a cluster must have at least 2 reviews
                clustering = DBSCAN(eps=0.3, min_samples=2, metric='precomputed').fit(distance_matrix)
                
                # Get cluster labels (-1 means noise/no cluster)
                cluster_labels = clustering.labels_
                
                # Count clusters (excluding noise)
                n_clusters = len(set([label for label in cluster_labels if label >= 0]))
                
                # If we have clusters, analyze them
                if n_clusters > 0:
                    # Track cluster data for this establishment
                    establishment_clusters = []
                    
                    # Get sizes of all clusters
                    cluster_sizes = {}
                    for label in range(n_clusters):
                        indices = [i for i, l in enumerate(cluster_labels) if l == label]
                        cluster_sizes[label] = len(indices)
                    
                    # Calculate IQR for cluster sizes
                    cluster_size_values = list(cluster_sizes.values())
                    q1_size = np.percentile(cluster_size_values, 25)
                    q3_size = np.percentile(cluster_size_values, 75)
                    iqr_size = q3_size - q1_size
                    upper_bound_size = q3_size + (1.5 * iqr_size)
                    
                    # Analyze each cluster
                    for cluster_id in range(n_clusters):
                        # Get indices of reviews in this cluster
                        cluster_indices = [i for i, label in enumerate(cluster_labels) if label == cluster_id]
                        
                        if len(cluster_indices) < 2:
                            continue
                        
                        # Get original indices in the combined_reviews dataframe
                        original_indices = [valid_indices[i] for i in cluster_indices]
                        
                        # Calculate mean similarity within cluster
                        cluster_similarities = []
                        for i in range(len(cluster_indices)):
                            for j in range(i+1, len(cluster_indices)):
                                cluster_similarities.append(similarity_matrix[cluster_indices[i], cluster_indices[j]])
                        
                        mean_cluster_similarity = np.mean(cluster_similarities) if cluster_similarities else 0
                        
                        # Check if cluster size is above the upper IQR bound
                        is_suspicious_cluster = cluster_sizes[cluster_id] > upper_bound_size
                        
                        # Calculate review-to-cluster similarities
                        review_similarities = []
                        for idx in cluster_indices:
                            # Calculate mean similarity of this review to all others in the cluster
                            similarities = []
                            for other_idx in cluster_indices:
                                if idx != other_idx:
                                    similarities.append(similarity_matrix[idx, other_idx])
                            
                            mean_similarity = np.mean(similarities) if similarities else 0
                            review_similarities.append({
                                'index': valid_indices[idx],
                                'mean_similarity': mean_similarity
                            })
                        
                        # Calculate IQR for review similarity scores
                        similarity_values = [r['mean_similarity'] for r in review_similarities]
                        q1_sim = np.percentile(similarity_values, 25)
                        q3_sim = np.percentile(similarity_values, 75)
                        iqr_sim = q3_sim - q1_sim
                        upper_bound_sim = q3_sim + (1.5 * iqr_sim)
                        
                        # Identify suspicious reviews (those with mean similarity above upper IQR bound)
                        suspicious_review_indices = []
                        for review in review_similarities:
                            if review['mean_similarity'] > upper_bound_sim:
                                suspicious_review_indices.append(review['index'])
                                suspicious_reviews_list.append(review['index'])
                        
                        # Store cluster data
                        cluster_data = {
                            'placeId': place_id,
                            'cluster_id': cluster_id,
                            'size': cluster_sizes[cluster_id],
                            'mean_similarity': mean_cluster_similarity,
                            'is_suspicious': is_suspicious_cluster,
                            'suspicious_reviews_count': len(suspicious_review_indices)
                        }
                        
                        establishment_clusters.append(cluster_data)
                        all_clusters_data.append(cluster_data)
                        
                        # Track suspicious clusters
                        if is_suspicious_cluster:
                            suspicious_clusters.append({
                                'placeId': place_id,
                                'cluster_id': cluster_id,
                                'size': cluster_sizes[cluster_id],
                                'affected_reviews': original_indices
                            })
            
            except Exception as e:
                print(f"Error in authenticity detection for place_id {place_id}: {str(e)}")
                continue
        
        # Initialize suspicion flags and weights in the combined_reviews dataframe
        self.combined_reviews['is_suspicious_review'] = False
        self.combined_reviews['is_in_suspicious_cluster'] = False
        self.combined_reviews['suspicion_weight'] = 1.0
        
        # Mark suspicious reviews based on similarity outliers
        if suspicious_reviews_list:
            self.combined_reviews.loc[suspicious_reviews_list, 'is_suspicious_review'] = True
        
        # Mark reviews in suspicious clusters
        for cluster in suspicious_clusters:
            self.combined_reviews.loc[cluster['affected_reviews'], 'is_in_suspicious_cluster'] = True
        
        # Apply suspicion weights according to the specified rules:
        # - Suspicious review: weight = 0.5
        # - In suspicious cluster but not suspicious review: weight = 0.85
        # - Review suspicion overrides cluster suspicion
        
        # First apply cluster suspicion
        cluster_suspicious_mask = (self.combined_reviews['is_in_suspicious_cluster'] == True) & (self.combined_reviews['is_suspicious_review'] == False)
        self.combined_reviews.loc[cluster_suspicious_mask, 'suspicion_weight'] = 0.85
        
        # Then apply review suspicion (overrides cluster suspicion)
        self.combined_reviews.loc[self.combined_reviews['is_suspicious_review'] == True, 'suspicion_weight'] = 0.5
        
        # Create cluster information dataframe for analysis
        if all_clusters_data:
            self.cluster_df = pd.DataFrame(all_clusters_data)
            
            # Count suspicious clusters and reviews
            suspicious_clusters_count = sum(self.cluster_df['is_suspicious'])
            suspicious_reviews_count = len(suspicious_reviews_list)
            affected_by_cluster_count = sum(self.combined_reviews['is_in_suspicious_cluster'])
            
            print(f"Identified {suspicious_clusters_count} suspicious clusters")
            print(f"Identified {suspicious_reviews_count} suspicious reviews based on similarity")
            print(f"Total reviews affected by suspicion: {affected_by_cluster_count + suspicious_reviews_count}")
        else:
            print("No clusters found in the data")

    def analyze_authenticity_clusters(self):
        """
        Analyze suspicious review clusters and provide insights.
        Must be called after detect_review_authenticity().
        """
        if not hasattr(self, 'cluster_df') or self.cluster_df is None or len(self.cluster_df) == 0:
            print("No cluster information available. Run detect_review_authenticity first.")
            return None

        print("Analyzing suspicious review clusters...")
        
        try:
            # Add validation for required columns
            required_columns = ['placeId', 'size', 'mean_similarity', 'is_suspicious']
            if not all(col in self.cluster_df.columns for col in required_columns):
                print("Missing required columns in cluster data")
                return None
            
            # Add establishment names for better readability
            self.cluster_df = pd.merge(
                self.cluster_df,
                self.establishments_df[['placeId', 'title']],
                on='placeId',
                how='left'
            ).rename(columns={'title': 'establishment_name'})
            
            # Group by establishment and calculate summary metrics
            establishment_summary = self.cluster_df.groupby(['placeId', 'establishment_name']).agg({
                'size': ['mean', 'median', 'max', 'std', 'count'],
                'mean_similarity': ['mean', 'max', 'min'],
                'is_suspicious': 'sum',
                'suspicious_reviews_count': 'sum'
            }).reset_index()
            
            # Flatten the MultiIndex columns
            establishment_summary.columns = [
                '_'.join(col).strip('_') for col in establishment_summary.columns.values
            ]
            
            # Calculate total reviews affected by suspicion for each establishment
            suspicion_counts = self.combined_reviews.groupby('placeId').agg({
                'is_suspicious_review': 'sum',
                'is_in_suspicious_cluster': 'sum',
                'placeId': 'count'
            }).rename(columns={'placeId': 'total_reviews'})
            
            # Validate merged dataframe
            if len(establishment_summary) == 0:
                print("No establishments found after merging metrics")
                return None
            
            establishment_summary = pd.merge(
                establishment_summary,
                suspicion_counts,
                on='placeId',
                how='left'
            )
            
            # Calculate suspicious review ratios
            establishment_summary['suspicious_review_ratio'] = (
                establishment_summary['is_suspicious_review'] / establishment_summary['total_reviews'] * 100
            ).round(2)
            
            establishment_summary['suspicious_cluster_ratio'] = (
                establishment_summary['is_in_suspicious_cluster'] / establishment_summary['total_reviews'] * 100
            ).round(2)
            
            # Calculate total suspicion impact
            establishment_summary['total_suspicion_impact'] = (
                (establishment_summary['is_suspicious_review'] * 0.5) + 
                (establishment_summary['is_in_suspicious_cluster'] - establishment_summary['is_suspicious_review']) * 0.15
            ) / establishment_summary['total_reviews']
            
            # Add authenticity risk level based on total suspicion impact
            def get_risk_level(impact):
                if impact > 0.25:
                    return 'Very High'
                elif impact > 0.15:
                    return 'High'
                elif impact > 0.1:
                    return 'Medium'
                elif impact > 0.05:
                    return 'Low'
                else:
                    return 'Very Low'
                
            establishment_summary['authenticity_risk'] = establishment_summary['total_suspicion_impact'].apply(get_risk_level)
            
            # Sort by suspicion impact for reporting
            self.establishment_authenticity = establishment_summary.sort_values('total_suspicion_impact', ascending=False)
            
            # Summary statistics
            high_risk_count = sum(self.establishment_authenticity['authenticity_risk'].isin(['High', 'Very High']))
            total_suspicious_reviews = self.establishment_authenticity['is_suspicious_review'].sum()
            total_suspicious_clusters = self.establishment_authenticity['is_suspicious_sum'].sum()
            
            print(f"Found {high_risk_count} establishments with high or very high authenticity risk")
            print(f"Total suspicious reviews across all establishments: {total_suspicious_reviews}")
            print(f"Total suspicious clusters across all establishments: {total_suspicious_clusters}")
            
            # Return the top 10 establishments with highest risk
            top_risk = self.establishment_authenticity.head(10)[
                ['establishment_name', 'total_reviews', 'size_count', 'is_suspicious_sum',
                 'suspicious_reviews_count_sum', 'suspicious_review_ratio', 'suspicious_cluster_ratio',
                 'total_suspicion_impact', 'authenticity_risk']
            ].rename(columns={
                'size_count': 'clusters_count',
                'is_suspicious_sum': 'suspicious_clusters',
                'suspicious_reviews_count_sum': 'suspicious_reviews'
            })
            
            print("\nTop 10 establishments with highest authenticity risk:")
            print(top_risk)
            
            return self.establishment_authenticity
            
        except Exception as e:
            print(f"Error in authenticity cluster analysis: {str(e)}")
            return None
    
    def prepare_review_labels(self):
        """
        Prepare specialized labels for reviews to support metric calculation:
        - Sentiment category
        - Price evaluation
        - Recommendation
        - Complaint and resolution flags
        
        Uses context-aware detection for complaints to handle negations.
        """
        print("Preparing review labels...")
        
        # Initialize label columns
        self.combined_reviews['sentiment_category'] = 'neutral'
        self.combined_reviews['has_price_evaluation'] = False
        self.combined_reviews['price_sentiment'] = 'neutral'
        self.combined_reviews['has_recommendation'] = False
        self.combined_reviews['recommendation_type'] = 'neutral'
        self.combined_reviews['is_complaint'] = False
        self.combined_reviews['has_response'] = False
        self.combined_reviews['has_resolution'] = False
        
        # Define negation words for context-aware detection
        negation_words = ['no', 'not', 'never', 'without', 'free of', 'didn\'t have', 'doesn\'t have', 
                          'didn\'t experience', 'solved', 'fixed', 'resolved', 'avoided', 'prevented']
        
        # Sentiment analysis for each review
        for idx, review in self.combined_reviews.iterrows():
            if pd.isna(review['review_text']) or review['review_text'] == '':
                continue
                
            text = review['review_text'].lower()
            
            # Calculate sentiment
            sentiment = self.sia.polarity_scores(text)
            
            # Categorize sentiment
            if sentiment['compound'] >= 0.05:
                self.combined_reviews.at[idx, 'sentiment_category'] = 'positive'
            elif sentiment['compound'] <= -0.05:
                self.combined_reviews.at[idx, 'sentiment_category'] = 'negative'
            
            # Check for price evaluation
            has_price_keyword = any(keyword in text for keyword in self.price_keywords['positive'] + self.price_keywords['negative'])
            
            if has_price_keyword:
                self.combined_reviews.at[idx, 'has_price_evaluation'] = True
                
                # Determine price sentiment
                positive_price = any(keyword in text for keyword in self.price_keywords['positive'])
                negative_price = any(keyword in text for keyword in self.price_keywords['negative'])
                
                if positive_price and not negative_price:
                    self.combined_reviews.at[idx, 'price_sentiment'] = 'positive'
                elif negative_price and not positive_price:
                    self.combined_reviews.at[idx, 'price_sentiment'] = 'negative'
            
            # Check for recommendation
            has_recommendation = any(keyword in text for keyword in 
                                    self.recommendation_keywords['positive'] + 
                                    self.recommendation_keywords['negative'])
            
            if has_recommendation:
                self.combined_reviews.at[idx, 'has_recommendation'] = True
                
                # Determine recommendation type
                positive_rec = any(keyword in text for keyword in self.recommendation_keywords['positive'])
                negative_rec = any(keyword in text for keyword in self.recommendation_keywords['negative'])
                
                if positive_rec and not negative_rec:
                    self.combined_reviews.at[idx, 'recommendation_type'] = 'positive'
                elif negative_rec and not positive_rec:
                    self.combined_reviews.at[idx, 'recommendation_type'] = 'negative'
            
            # Context-aware complaint detection
            is_complaint = False
            
            # Split into sentences for better context analysis
            sentences = re.split(r'[.!?]', text)
            
            for sentence in sentences:
                sentence = sentence.strip().lower()
                if not sentence:
                    continue
                
                # Check if sentence contains a complaint keyword
                contains_complaint_keyword = any(keyword in sentence for keyword in self.complaint_keywords)
                
                if contains_complaint_keyword:
                    # Check for negation in the same sentence
                    has_negation = any(neg in sentence for neg in negation_words)
                    
                    # Check sentence sentiment
                    sentence_sentiment = self.sia.polarity_scores(sentence)['compound']
                    
                    # Consider it a complaint if:
                    # 1. It contains a complaint keyword AND
                    # 2. It does NOT contain negation words AND
                    # 3. The sentence sentiment is negative or neutral
                    if not has_negation and sentence_sentiment <= 0.1:
                        is_complaint = True
                        break
            
            # If overall review sentiment is very negative, also consider it a complaint
            if sentiment['compound'] <= -0.4:  # Strong negative sentiment
                is_complaint = True
            
            self.combined_reviews.at[idx, 'is_complaint'] = is_complaint
            
            if is_complaint:
                # Check for response
                has_response = not pd.isna(review['response_text']) and review['response_text'] != ''
                self.combined_reviews.at[idx, 'has_response'] = has_response
                
                if has_response:
                    # Check for resolution language
                    response_text = review['response_text'].lower()
                    has_resolution = any(keyword in response_text for keyword in self.resolution_keywords)
                    self.combined_reviews.at[idx, 'has_resolution'] = has_resolution
        
        # Calculate total weight for each review (combining time, local guide, and suspicion)
        self.combined_reviews['total_weight'] = (
            self.combined_reviews['time_weight'] * 
            self.combined_reviews['local_guide_weight'] * 
            self.combined_reviews['suspicion_weight']
        )
        
        # Print some statistics
        print(f"Positive reviews: {(self.combined_reviews['sentiment_category'] == 'positive').sum()}")
        print(f"Negative reviews: {(self.combined_reviews['sentiment_category'] == 'negative').sum()}")
        print(f"Complaint reviews: {self.combined_reviews['is_complaint'].sum()}")
    
    def calculate_weighted_rating_score(self):
        """Calculate the weighted rating score for each establishment."""
        print("Calculating Rating Score...")
        
        rating_metrics = []
        
        for place_id in self.combined_reviews['placeId'].unique():
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            if len(place_reviews) == 0:
                continue
            
            # Calculate weighted average rating
            weighted_sum = (place_reviews['rating'] * place_reviews['total_weight']).sum()
            total_weights = place_reviews['total_weight'].sum()
            
            if total_weights > 0:
                weighted_avg_rating = weighted_sum / total_weights
            else:
                weighted_avg_rating = place_reviews['rating'].mean()  # Fallback to regular mean
            
            # Store metrics
            establishment_name = self.establishments_df.loc[
                self.establishments_df['placeId'] == place_id, 'title'
            ].iloc[0] if not self.establishments_df[self.establishments_df['placeId'] == place_id].empty else ""
            
            rating_metrics.append({
                'placeId': place_id,
                'establishment_name': establishment_name,
                'total_reviews': len(place_reviews),
                'weighted_avg_rating': weighted_avg_rating,
                'raw_avg_rating': place_reviews['rating'].mean()
            })
        
        self.rating_metrics_df = pd.DataFrame(rating_metrics)
        
        # Min-max normalization for rating score
        if not self.rating_metrics_df.empty:
            min_rating = self.rating_metrics_df['weighted_avg_rating'].min()
            max_rating = self.rating_metrics_df['weighted_avg_rating'].max()
            
            if max_rating > min_rating:
                self.rating_metrics_df['rating_score'] = (
                    (self.rating_metrics_df['weighted_avg_rating'] - min_rating) / (max_rating - min_rating)
                )
            else:
                self.rating_metrics_df['rating_score'] = 0.5
        
        print(f"Rating Score calculated for {len(self.rating_metrics_df)} establishments")
        return self.rating_metrics_df
    
    def calculate_volume_score(self):
        """Calculate the volume score based on the number of weighted reviews."""
        print("Calculating Volume Score...")
        
        volume_metrics = []
        
        for place_id in self.combined_reviews['placeId'].unique():
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            if len(place_reviews) == 0:
                continue
            
            # Calculate weighted review count
            weighted_count = place_reviews['total_weight'].sum()
            
            # Count by source
            google_count = place_reviews[place_reviews['source'] == 'Google Maps'].shape[0]
            trustpilot_count = place_reviews[place_reviews['source'] == 'Trustpilot'].shape[0]
            
            # Store metrics
            establishment_name = self.establishments_df.loc[
                self.establishments_df['placeId'] == place_id, 'title'
            ].iloc[0] if not self.establishments_df[self.establishments_df['placeId'] == place_id].empty else ""
            
            volume_metrics.append({
                'placeId': place_id,
                'establishment_name': establishment_name,
                'total_reviews': len(place_reviews),
                'weighted_volume': weighted_count,
                'google_count': google_count,
                'trustpilot_count': trustpilot_count
            })
        
        self.volume_metrics_df = pd.DataFrame(volume_metrics)
        
        # Min-max normalization for volume score
        if not self.volume_metrics_df.empty:
            min_volume = self.volume_metrics_df['weighted_volume'].min()
            max_volume = self.volume_metrics_df['weighted_volume'].max()
            
            if max_volume > min_volume:
                self.volume_metrics_df['volume_score'] = (
                    (self.volume_metrics_df['weighted_volume'] - min_volume) / (max_volume - min_volume)
                )
            else:
                self.volume_metrics_df['volume_score'] = 0.5
        
        print(f"Volume Score calculated for {len(self.volume_metrics_df)} establishments")
        return self.volume_metrics_df
    
    def calculate_communication_score(self):
        """Calculate the communication score based on complaint handling and response quality."""
        print("Calculating Communication Score...")
        
        communication_metrics = []
        
        for place_id in self.combined_reviews['placeId'].unique():
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            if len(place_reviews) == 0:
                continue
            
            # Get complaint reviews
            complaint_reviews = place_reviews[place_reviews['is_complaint'] == True]
            
            # Calculate weighted metrics
            total_reviews = len(place_reviews)
            complaint_count = len(complaint_reviews)
            
            # Calculate weighted communication metrics
            weighted_complaints = complaint_reviews['total_weight'].sum()
            weighted_responses = complaint_reviews[complaint_reviews['has_response']]['total_weight'].sum()
            weighted_resolutions = complaint_reviews[complaint_reviews['has_resolution']]['total_weight'].sum()
            
            # Calculate communication score using updated formula
            if weighted_complaints > 0:
                # Percentage of total reviews that are complaints
                complaint_pct = weighted_complaints / place_reviews['total_weight'].sum()
                
                # Percentage of complaints with only response (no resolution)
                only_response_pct = ((weighted_responses - weighted_resolutions) / weighted_complaints) if weighted_complaints > 0 else 0
                
                # Percentage of complaints with resolution
                resolution_pct = weighted_resolutions / weighted_complaints if weighted_complaints > 0 else 0
                
                # Calculate raw communication score (higher means better communication)
                raw_communication_score = 1 - (complaint_pct - (only_response_pct * 0.5) - (resolution_pct * 0.75))
            else:
                # No complaints is the best possible score
                raw_communication_score = 1.0
            
            # Store metrics
            establishment_name = self.establishments_df.loc[
                self.establishments_df['placeId'] == place_id, 'title'
            ].iloc[0] if not self.establishments_df[self.establishments_df['placeId'] == place_id].empty else ""
            
            communication_metrics.append({
                'placeId': place_id,
                'establishment_name': establishment_name,
                'total_reviews': total_reviews,
                'complaint_count': complaint_count,
                'weighted_complaints': weighted_complaints,
                'weighted_responses': weighted_responses,
                'weighted_resolutions': weighted_resolutions,
                'raw_communication_score': raw_communication_score
            })
        
        self.communication_metrics_df = pd.DataFrame(communication_metrics)
        
        # Min-max normalization for communication score
        if not self.communication_metrics_df.empty:
            min_score = self.communication_metrics_df['raw_communication_score'].min()
            max_score = self.communication_metrics_df['raw_communication_score'].max()
            
            if max_score > min_score:
                self.communication_metrics_df['communication_score'] = (
                    (self.communication_metrics_df['raw_communication_score'] - min_score) / (max_score - min_score)
                )
            else:
                self.communication_metrics_df['communication_score'] = 0.5
        
        print(f"Communication Score calculated for {len(self.communication_metrics_df)} establishments")
        return self.communication_metrics_df
    
    def calculate_service_score(self):
        """Calculate the service score based on positive vs negative reviews."""
        print("Calculating Service Score...")
        
        service_metrics = []
        
        for place_id in self.combined_reviews['placeId'].unique():
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            if len(place_reviews) == 0:
                continue
            
            # Apply weights to positive and negative reviews
            positive_reviews = place_reviews[place_reviews['sentiment_category'] == 'positive']
            negative_reviews = place_reviews[place_reviews['sentiment_category'] == 'negative']
            
            # Calculate weighted sentiment balance
            weighted_positive = positive_reviews['total_weight'].sum()
            weighted_negative = negative_reviews['total_weight'].sum()
            total_weighted = place_reviews['total_weight'].sum()
            
            if total_weighted > 0:
                raw_service_score = (weighted_positive - weighted_negative) / total_weighted
            else:
                raw_service_score = 0
            
            # Store metrics
            establishment_name = self.establishments_df.loc[
                self.establishments_df['placeId'] == place_id, 'title'
            ].iloc[0] if not self.establishments_df[self.establishments_df['placeId'] == place_id].empty else ""
            
            service_metrics.append({
                'placeId': place_id,
                'establishment_name': establishment_name,
                'total_reviews': len(place_reviews),
                'positive_count': len(positive_reviews),
                'negative_count': len(negative_reviews),
                'weighted_positive': weighted_positive,
                'weighted_negative': weighted_negative,
                'raw_service_score': raw_service_score
            })
        
        self.service_metrics_df = pd.DataFrame(service_metrics)
        
        # Min-max normalization for service score
        if not self.service_metrics_df.empty:
            min_score = self.service_metrics_df['raw_service_score'].min()
            max_score = self.service_metrics_df['raw_service_score'].max()
            
            if max_score > min_score:
                self.service_metrics_df['service_score'] = (
                    (self.service_metrics_df['raw_service_score'] - min_score) / (max_score - min_score)
                )
            else:
                self.service_metrics_df['service_score'] = 0.5
        
        print(f"Service Score calculated for {len(self.service_metrics_df)} establishments")
        return self.service_metrics_df
    
    def calculate_affordability_score(self):
        """Calculate the affordability score based on price evaluations in reviews."""
        print("Calculating Affordability Score...")
        
        affordability_metrics = []
        
        for place_id in self.combined_reviews['placeId'].unique():
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            if len(place_reviews) == 0:
                continue
            
            # Get reviews with price evaluation
            price_reviews = place_reviews[place_reviews['has_price_evaluation'] == True]
            
            if len(price_reviews) == 0:
                # No price evaluations available
                raw_affordability_score = 0.5  # Neutral default score
            else:
                # Calculate weighted price sentiment
                positive_price = price_reviews[price_reviews['price_sentiment'] == 'positive']
                negative_price = price_reviews[price_reviews['price_sentiment'] == 'negative']
                
                weighted_positive = positive_price['total_weight'].sum()
                weighted_negative = negative_price['total_weight'].sum()
                total_weighted = price_reviews['total_weight'].sum()
                
                if total_weighted > 0:
                    raw_affordability_score = (weighted_positive - weighted_negative) / total_weighted
                else:
                    raw_affordability_score = 0
            
            # Store metrics
            establishment_name = self.establishments_df.loc[
                self.establishments_df['placeId'] == place_id, 'title'
            ].iloc[0] if not self.establishments_df[self.establishments_df['placeId'] == place_id].empty else ""
            
            affordability_metrics.append({
                'placeId': place_id,
                'establishment_name': establishment_name,
                'total_reviews': len(place_reviews),
                'price_evaluation_count': len(price_reviews),
                'positive_price_count': len(positive_price) if 'positive_price' in locals() else 0,
                'negative_price_count': len(negative_price) if 'negative_price' in locals() else 0,
                'raw_affordability_score': raw_affordability_score
            })
        
        self.affordability_metrics_df = pd.DataFrame(affordability_metrics)
        
        # Min-max normalization for affordability score
        if not self.affordability_metrics_df.empty:
            min_score = self.affordability_metrics_df['raw_affordability_score'].min()
            max_score = self.affordability_metrics_df['raw_affordability_score'].max()
            
            if max_score > min_score:
                self.affordability_metrics_df['affordability_score'] = (
                    (self.affordability_metrics_df['raw_affordability_score'] - min_score) / (max_score - min_score)
                )
            else:
                self.affordability_metrics_df['affordability_score'] = 0.5
        
        print(f"Affordability Score calculated for {len(self.affordability_metrics_df)} establishments")
        return self.affordability_metrics_df
    
    def calculate_recommendation_score(self):
        """Calculate the recommendation score based on explicit recommendations in reviews."""
        print("Calculating Recommendation Score...")
        
        recommendation_metrics = []
        
        for place_id in self.combined_reviews['placeId'].unique():
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            if len(place_reviews) == 0:
                continue
            
            # Get reviews with recommendations
            rec_reviews = place_reviews[place_reviews['has_recommendation'] == True]
            
            if len(rec_reviews) == 0:
                # No recommendations available
                raw_recommendation_score = 0.5  # Neutral default score
            else:
                # Calculate weighted recommendation sentiment
                positive_rec = rec_reviews[rec_reviews['recommendation_type'] == 'positive']
                negative_rec = rec_reviews[rec_reviews['recommendation_type'] == 'negative']
                
                weighted_positive = positive_rec['total_weight'].sum()
                weighted_negative = negative_rec['total_weight'].sum()
                total_weighted = rec_reviews['total_weight'].sum()
                
                if total_weighted > 0:
                    raw_recommendation_score = (weighted_positive - weighted_negative) / total_weighted
                else:
                    raw_recommendation_score = 0
            
            # Store metrics
            establishment_name = self.establishments_df.loc[
                self.establishments_df['placeId'] == place_id, 'title'
            ].iloc[0] if not self.establishments_df[self.establishments_df['placeId'] == place_id].empty else ""
            
            recommendation_metrics.append({
                'placeId': place_id,
                'establishment_name': establishment_name,
                'total_reviews': len(place_reviews),
                'recommendation_count': len(rec_reviews),
                'positive_rec_count': len(positive_rec) if 'positive_rec' in locals() else 0,
                'negative_rec_count': len(negative_rec) if 'negative_rec' in locals() else 0,
                'raw_recommendation_score': raw_recommendation_score
            })
        
        self.recommendation_metrics_df = pd.DataFrame(recommendation_metrics)
        
        # Min-max normalization for recommendation score
        if not self.recommendation_metrics_df.empty:
            min_score = self.recommendation_metrics_df['raw_recommendation_score'].min()
            max_score = self.recommendation_metrics_df['raw_recommendation_score'].max()
            
            if max_score > min_score:
                self.recommendation_metrics_df['recommendation_score'] = (
                    (self.recommendation_metrics_df['raw_recommendation_score'] - min_score) / (max_score - min_score)
                )
            else:
                self.recommendation_metrics_df['recommendation_score'] = 0.5
        
        print(f"Recommendation Score calculated for {len(self.recommendation_metrics_df)} establishments")
        return self.recommendation_metrics_df
    
    def calculate_composite_score(self):
        """
        Calculate the final composite score combining all individual scores using the specified formula:
        final_score = 
            (rating_score × 0.5) 
          + volume_score × 0.15 
          + (communication_score × 0.25 
             + service_score × 0.15 
             + affordability_score × 0.15 
             + recommendation_score × 0.15) × 0.35
        """
        print("Calculating Composite Score...")
        
        # Ensure all necessary metrics are calculated
        if not hasattr(self, 'rating_metrics_df'):
            self.calculate_weighted_rating_score()
        
        if not hasattr(self, 'volume_metrics_df'):
            self.calculate_volume_score()
        
        if not hasattr(self, 'communication_metrics_df'):
            self.calculate_communication_score()
        
        if not hasattr(self, 'service_metrics_df'):
            self.calculate_service_score()
        
        if not hasattr(self, 'affordability_metrics_df'):
            self.calculate_affordability_score()
        
        if not hasattr(self, 'recommendation_metrics_df'):
            self.calculate_recommendation_score()
        
        # Create a base DataFrame with all establishments
        all_place_ids = set(self.combined_reviews['placeId'].unique())
        
        # Create a DataFrame with all metrics
        composite_df = pd.DataFrame({'placeId': list(all_place_ids)})
        
        # Merge with all metric DataFrames
        composite_df = pd.merge(composite_df, self.rating_metrics_df[['placeId', 'establishment_name', 'rating_score']], 
                               on='placeId', how='left')
        
        composite_df = pd.merge(composite_df, self.volume_metrics_df[['placeId', 'volume_score']], 
                               on='placeId', how='left')
        
        composite_df = pd.merge(composite_df, self.communication_metrics_df[['placeId', 'communication_score']], 
                               on='placeId', how='left')
        
        composite_df = pd.merge(composite_df, self.service_metrics_df[['placeId', 'service_score']], 
                               on='placeId', how='left')
        
        composite_df = pd.merge(composite_df, self.affordability_metrics_df[['placeId', 'affordability_score']], 
                               on='placeId', how='left')
        
        composite_df = pd.merge(composite_df, self.recommendation_metrics_df[['placeId', 'recommendation_score']], 
                               on='placeId', how='left')
        
        # Fill missing values with neutral scores
        for col in ['rating_score', 'volume_score', 'communication_score', 
                   'service_score', 'affordability_score', 'recommendation_score']:
            composite_df[col].fillna(0.5, inplace=True)
        
        # Calculate the composite score using the formula
        composite_df['other_factors_score'] = (
            composite_df['communication_score'] * 0.25 +
            composite_df['service_score'] * 0.15 +
            composite_df['affordability_score'] * 0.15 +
            composite_df['recommendation_score'] * 0.15
        )
        
        composite_df['composite_score'] = (
            composite_df['rating_score'] * 0.5 +
            composite_df['volume_score'] * 0.15 +
            composite_df['other_factors_score'] * 0.35
        )
        
        # Scale to 0-100 for readability
        composite_df['final_score'] = composite_df['composite_score'] * 100
        
        # Add rank
        composite_df['rank'] = composite_df['final_score'].rank(ascending=False, method='min').astype(int)
        
        # Add additional metrics for reference
        composite_df = pd.merge(composite_df, 
                               self.rating_metrics_df[['placeId', 'total_reviews', 'weighted_avg_rating', 'raw_avg_rating']], 
                               on='placeId', how='left')
        
        # Keep only necessary columns
        self.results_df = composite_df[
            ['placeId', 'establishment_name', 'rank', 'final_score', 
             'total_reviews', 'weighted_avg_rating', 'raw_avg_rating',
             'rating_score', 'volume_score', 'communication_score', 
             'service_score', 'affordability_score', 'recommendation_score']
        ]
        
        # Sort by rank
        self.results_df = self.results_df.sort_values('rank')
        
        print(f"Composite Score calculated for {len(self.results_df)} establishments")
        return self.results_df
    
    def filter_establishments_by_review_count(self):
        """
        Filter out establishments that have fewer reviews than the minimum threshold.
        Updates the combined_reviews DataFrame to only include establishments that meet the threshold.
        """
        print("Filtering establishments by review count...")
        
        # Get the minimum review threshold from config
        min_reviews_threshold = self.config['analysis']['filtering']['min_reviews_threshold']
        
        # Count reviews per establishment
        review_counts = self.combined_reviews['placeId'].value_counts()
        
        # Get establishments that meet the threshold
        valid_establishments = review_counts[review_counts >= min_reviews_threshold].index
        
        # Filter the combined_reviews DataFrame
        initial_count = len(self.combined_reviews)
        self.combined_reviews = self.combined_reviews[self.combined_reviews['placeId'].isin(valid_establishments)]
        
        # Print statistics
        removed_count = initial_count - len(self.combined_reviews)
        print(f"Removed {removed_count} reviews from establishments with fewer than {min_reviews_threshold} reviews")
        print(f"Remaining reviews: {len(self.combined_reviews)}")
        print(f"Remaining establishments: {len(valid_establishments)}")

    def run_analysis(self):
        """Run the full enhanced analysis pipeline."""
        self.detect_review_authenticity()
        # Analyze authenticity clusters
        self.analyze_authenticity_clusters()
        
        # Filter establishments by review count
        self.filter_establishments_by_review_count()
        
        # Calculate all required metrics
        self.calculate_weighted_rating_score()
        self.calculate_volume_score()
        self.calculate_communication_score()
        self.calculate_service_score()
        self.calculate_affordability_score()
        self.calculate_recommendation_score()
        
        # Calculate final composite score
        return self.calculate_composite_score()

    
    def save_results(self, output_dir='.'):
        """Save analysis results to Excel files."""
        print("Saving results...")

        # Get current date and time for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results
        if self.results_df is not None:
            self.results_df.to_excel(f"{output_dir}/establishment_rankings_{timestamp}.xlsx", index=False)
            print(f"Enhanced rankings saved to {output_dir}/establishment_rankings_{timestamp}.xlsx")
        
        # Save detailed metrics for reference
        metrics_df = pd.DataFrame({'placeId': self.results_df['placeId'].unique()})
        
        # Add all metric details
        metrics_df = pd.merge(metrics_df, self.rating_metrics_df, on='placeId', how='left')
        metrics_df = pd.merge(metrics_df, self.volume_metrics_df[['placeId', 'weighted_volume', 'google_count', 
                                                                 'trustpilot_count', 'volume_score']], 
                             on='placeId', how='left')
        metrics_df = pd.merge(metrics_df, self.communication_metrics_df[['placeId', 'complaint_count', 'weighted_complaints', 
                                                                    'weighted_responses', 'weighted_resolutions', 
                                                                    'raw_communication_score', 'communication_score']], 
                             on='placeId', how='left')
        metrics_df = pd.merge(metrics_df, self.service_metrics_df[['placeId', 'positive_count', 'negative_count',
                                                                 'raw_service_score', 'service_score']], 
                             on='placeId', how='left')
        metrics_df = pd.merge(metrics_df, self.affordability_metrics_df[['placeId', 'price_evaluation_count',
                                                                       'raw_affordability_score', 'affordability_score']], 
                             on='placeId', how='left')
        metrics_df = pd.merge(metrics_df, self.recommendation_metrics_df[['placeId', 'recommendation_count',
                                                                        'raw_recommendation_score', 'recommendation_score']], 
                             on='placeId', how='left')
        
        # Add final score and rank
        metrics_df = pd.merge(metrics_df, self.results_df[['placeId', 'final_score', 'rank']], 
                             on='placeId', how='left')
        
        # Save detailed metrics
        metrics_df.to_excel(f"{output_dir}/establishment_detailed_metrics_{timestamp}.xlsx", index=False)
        print(f"Detailed metrics saved to {output_dir}/establishment_detailed_metrics_{timestamp}.xlsx")
        
        print("All results saved successfully")

    def update_clinics_cms_template(self, analysis_dir='analysis_results', template_path='website_uploads/clinics_cms_template.csv'):
        """
        Creates a new clinics CMS file based on the latest establishment rankings.
        
        Parameters:
        -----------
        analysis_dir : str
            Directory containing the analysis results
        template_path : str
            Path to the clinics CMS template CSV file (used as schema reference)
        """
        print("Creating new clinics CMS file from latest rankings...")
        
        # Get the latest establishment rankings file
        ranking_files = [f for f in os.listdir(analysis_dir) if f.startswith('establishment_rankings_') and f.endswith('.xlsx')]
        if not ranking_files:
            print("No establishment rankings files found in the specified directory")
            return
            
        # Sort files by modification time to get the latest
        latest_ranking = max(ranking_files, key=lambda x: os.path.getmtime(os.path.join(analysis_dir, x)))
        ranking_path = os.path.join(analysis_dir, latest_ranking)
        
        # Load the rankings and template
        rankings_df = pd.read_excel(ranking_path)
        template_df = pd.read_csv(template_path)
        
        # Create a new DataFrame with the template schema
        new_cms_df = pd.DataFrame(columns=template_df.columns)
        
        # Fill in the data from rankings
        for _, row in rankings_df.iterrows():
            # Find matching establishment in template (if exists)
            template_match = template_df[template_df['Name'].str.lower() == row['establishment_name'].lower()]
            
            # Create new row with template schema
            new_row = {
                'Name': row['establishment_name'],
                'Slug': template_match['Slug'].iloc[0] if not template_match.empty else row['establishment_name'].lower().replace(' ', '-'),
                'Collection ID': template_match['Collection ID'].iloc[0] if not template_match.empty else '',
                'Locale ID': template_match['Locale ID'].iloc[0] if not template_match.empty else '',
                'Item ID': template_match['Item ID'].iloc[0] if not template_match.empty else '',
                'Created On': template_match['Created On'].iloc[0] if not template_match.empty else datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Updated On': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Published On': template_match['Published On'].iloc[0] if not template_match.empty else datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                'Establishment Picture': template_match['Establishment Picture'].iloc[0] if not template_match.empty else '',
                'Rank': row['rank'],
                'Score': row['final_score'],
                'Total Reviews': row['total_reviews'],
                'Weighted Average Rating': row['weighted_avg_rating'],
                'Communication Score': row['communication_score']*100,
                'Service Score': row['service_score']*100,
                'Affordability Score': row['affordability_score']*100,
                'Recommendation Score': row['recommendation_score']*100,
                'Website': template_match['Website'].iloc[0] if not template_match.empty else '',
                'E-mail': template_match['E-mail'].iloc[0] if not template_match.empty else '',
                'Phone': template_match['Phone'].iloc[0] if not template_match.empty else ''
            }
            
            new_cms_df = pd.concat([new_cms_df, pd.DataFrame([new_row])], ignore_index=True)
        
        # Generate output filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = os.path.join(os.path.dirname(template_path), f'clinics_cms_{timestamp}.csv')
        
        # Save the new CMS file
        new_cms_df.to_csv(output_path, index=False)
        print(f"New clinics CMS file created successfully: {output_path}")
        print(f"Based on rankings from: {latest_ranking}")


# Example usage
if __name__ == "__main__":
    # File paths
    google_maps_file = "reviews/google/allGoogleReviews_2025-04-06.xlsx"
    trustpilot_file = "reviews/trustpilot/allTrustpilotReviews_2025-04-06.xlsx"
    establishment_file = "establishments/establishment_base.xlsx"
    
    # Create analyzer
    analyzer = EnhancedReviewAnalyzer(google_maps_file, trustpilot_file, establishment_file)
    
    # Run full analysis
    results = analyzer.run_analysis()
    
    # Save results
    analyzer.save_results("results")
    
    # Print top 10 establishments
    print("\nTop 10 Establishments:")
    print(results.head(10)[['rank', 'establishment_name', 'final_score', 'weighted_avg_rating', 'total_reviews']])