# version: 1.2.0
# TODO: 
# -Increase local guide weights
# -Add complaint resolution to composite score calculation
# -Move composite score weights to config file

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

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

def load_config():
    with open('config.yaml', 'r') as file:
        return yaml.safe_load(file)

# Load spaCy model for NER and language processing
nlp = spacy.load('en_core_web_sm')

# Suppress warnings
warnings.filterwarnings('ignore')

class ReviewAnalyzer:
    def __init__(self, google_maps_file, trustpilot_file, establishment_file):
        """
        Initialize the ReviewAnalyzer with the paths to the data files.
        
        Parameters:
        -----------
        google_maps_file : str
            Path to the Google Maps reviews CSV file
        trustpilot_file : str
            Path to the Trustpilot reviews CSV file
        establishment_file : str
            Path to the establishment base data CSV file
        """
        self.config = load_config()["analysis"]
        self.google_maps_file = google_maps_file
        self.trustpilot_file = trustpilot_file
        self.establishment_file = establishment_file
        
        # Initialize sentiment analyzer
        self.sia = SentimentIntensityAnalyzer()
        
        # Define business aspects for aspect-based sentiment analysis
        self.aspects = {
            'service': ['service', 'staff', 'employee', 'waiter', 'waitress', 'server', 'customer service'],
            'quality': ['quality', 'excellent', 'great', 'good', 'bad', 'poor', 'terrible'],
            'price': ['price', 'cost', 'expensive', 'cheap', 'affordable', 'value', 'money'],
            'ambiance': ['ambiance', 'atmosphere', 'environment', 'decor', 'music', 'noise', 'quiet'],
            'cleanliness': ['clean', 'dirty', 'filthy', 'hygiene', 'sanitary', 'neat', 'tidy'],
            'location': ['location', 'area', 'neighborhood', 'parking', 'accessible', 'central']
        }
        
        # Define time weights for recency
        self.time_weights = {
            30: 1.0,    # Last month: full weight
            90: 0.95,    # Last quarter: 90%
            180: 0.85,   # Last 6 months: 80%
            365: 0.8,   # Last year: 70%
            730: 0.6,   # Last 2 years: 50%
            1095: 0.3,  # Last 3 years: 30%
            float('inf'): 0.1  # Older: 10%
        }
        
        # Load and preprocess data
        self.load_data()
        self.preprocess_data()
        
    def load_data(self):
        """Load the data files into pandas DataFrames."""
        print("Loading data...")
        
        # Load Google Maps reviews
        self.google_df = pd.read_excel(self.google_maps_file)
        
        # Load Trustpilot reviews
        self.trustpilot_df = pd.read_excel(self.trustpilot_file)
        
        # Load establishment data (this remains CSV)
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
        self.google_df['responseFromOwnerText'].fillna('', inplace=True)
        self.trustpilot_df['replyMessage'].fillna('', inplace=True)
        
        # Create combined review dataset
        self.process_combined_reviews()

        # Filter out reviews unrelated to hair transplantation
        self.filter_hair_transplant_reviews()
        print("Data preprocessing completed")
        
    def process_combined_reviews(self):
        """Create a combined review dataset with normalized fields."""
        # Extract relevant fields from Google Maps
        google_reviews = self.google_df[['placeId', 'stars', 'publishedAtDate', 
                                       'reviewerNumberOfReviews', 'isLocalGuide',
                                       'review_text', 'responseFromOwnerText_en', 
                                       'responseFromOwnerDate']].copy()
        
        google_reviews['source'] = 'Google Maps'
        google_reviews['rating'] = google_reviews['stars']
        google_reviews['review_date'] = pd.to_datetime(google_reviews['publishedAtDate']).dt.tz_localize(None)  # Make timezone-naive
        google_reviews['response_text'] = google_reviews['responseFromOwnerText_en']
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
        
        print(f"Combined dataset created with {len(self.combined_reviews)} reviews")
    
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

    def calculate_weighted_average(self, values, weights):
        """Calculate weighted average of values based on weights."""
        return sum(v * w for v, w in zip(values, weights)) / sum(weights)

    def calculate_basic_metrics(self):
        """Calculate basic metrics with higher weights for local guides."""
        print("Calculating basic metrics...")
        
        # Set the weight multiplier for local guides
        local_guide_weight = 1.2  # Local guides get 20% more weight
        
        metrics = []
        
        for place_id, establishment in self.establishments_df.iterrows():
            place_id = establishment['placeId']
            
            # Get reviews for this establishment
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            google_reviews = place_reviews[place_reviews['source'] == 'Google Maps']
            trustpilot_reviews = place_reviews[place_reviews['source'] == 'Trustpilot']
            
            # Skip if no reviews
            if len(place_reviews) == 0:
                continue
                
            # Basic count metrics
            total_reviews = len(place_reviews)
            google_review_count = len(google_reviews)
            trustpilot_review_count = len(trustpilot_reviews)

            # Apply weights based on local guide status
            weights = [local_guide_weight if is_guide else 1.0 
                  for is_guide in place_reviews['isLocalGuide']]
            
            # Rating metrics
            avg_rating = self.calculate_weighted_average(place_reviews['rating'], weights)
            google_avg_rating = google_reviews['rating'].mean() if len(google_reviews) > 0 else np.nan
            trustpilot_avg_rating = trustpilot_reviews['rating'].mean() if len(trustpilot_reviews) > 0 else np.nan
            rating_std = np.sqrt(self.calculate_weighted_average((place_reviews['rating'] - avg_rating)**2, weights))
            
            # Rating distribution (percentage in each star category)
            rating_dist = {}
            for i in range(1, 6):
                rating_dist[f'rating_{i}_pct'] = (place_reviews['rating'] == i).mean() * 100
                
            # Cross-platform consistency
            if not np.isnan(google_avg_rating) and not np.isnan(trustpilot_avg_rating):
                rating_consistency = 1 - min(abs(google_avg_rating - trustpilot_avg_rating) / 4, 1)
            else:
                rating_consistency = np.nan
                
            # Recency metrics
            avg_days_since_review = place_reviews['days_since_review'].mean()
            # Percentage of reviews in the last 90 days
            recent_reviews = place_reviews[place_reviews['days_since_review'] <= 90].shape[0]
            recent_reviews_pct = recent_reviews / total_reviews * 100 if total_reviews > 0 else 0
            
            # Response metrics
            has_response = ~place_reviews['response_text'].isna() & (place_reviews['response_text'] != '')
            response_rate = has_response.mean() * 100
            avg_response_time = place_reviews['response_time_days'].mean()
            
            # Review quality metrics
            avg_review_length = place_reviews['review_length'].mean()
            verified_reviewer_pct = place_reviews['isLocalGuide'].mean() * 100
            avg_reviewer_exp = place_reviews['reviewer_experience'].mean()
            
            # Create metrics dictionary
            place_metrics = {
                'placeId': place_id,
                'title': establishment['title'],
                'total_reviews': total_reviews,
                'google_review_count': google_review_count,
                'trustpilot_review_count': trustpilot_review_count,
                'avg_rating': avg_rating,
                'google_avg_rating': google_avg_rating,
                'trustpilot_avg_rating': trustpilot_avg_rating,
                'rating_std': rating_std,
                'rating_consistency': rating_consistency,
                'avg_days_since_review': avg_days_since_review,
                'recent_reviews_pct': recent_reviews_pct,
                'response_rate': response_rate,
                'avg_response_time': avg_response_time,
                'avg_review_length': avg_review_length,
                'verified_reviewer_pct': verified_reviewer_pct,
                'avg_reviewer_exp': avg_reviewer_exp,
            }
            
            # Add rating distribution
            place_metrics.update(rating_dist)
            
            metrics.append(place_metrics)
            
        self.metrics_df = pd.DataFrame(metrics)
        print(f"Basic metrics calculated for {len(self.metrics_df)} establishments")
        return self.metrics_df
        
    def calculate_sentiment_metrics(self):
        """Calculate sentiment metrics with higher weights for local guides."""
        print("Calculating sentiment metrics...")
        
        # Create placeholders for sentiment data
        self.combined_reviews['sentiment_score'] = np.nan
        self.combined_reviews['sentiment_magnitude'] = np.nan
        self.combined_reviews['sentiment_category'] = ''
        
        # Calculate sentiment for each review
        for idx, review in self.combined_reviews.iterrows():
            if pd.isna(review['review_text']) or review['review_text'] == '':
                continue
                
            # Get sentiment score using VADER
            sentiment = self.sia.polarity_scores(review['review_text'])
            self.combined_reviews.at[idx, 'sentiment_score'] = sentiment['compound']
            self.combined_reviews.at[idx, 'sentiment_magnitude'] = abs(sentiment['compound'])
            
            # Categorize sentiment
            if sentiment['compound'] >= 0.05:
                self.combined_reviews.at[idx, 'sentiment_category'] = 'positive'
            elif sentiment['compound'] <= -0.05:
                self.combined_reviews.at[idx, 'sentiment_category'] = 'negative'
            else:
                self.combined_reviews.at[idx, 'sentiment_category'] = 'neutral'
        
        # Set the weight multiplier for local guides
        local_guide_weight = 1.2  # Local guides get 20% more weight
        
        sentiment_metrics = []
        
        for place_id in self.metrics_df['placeId']:
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            if len(place_reviews) == 0:
                continue
                
            # Apply weights based on local guide status
            weights = [local_guide_weight if is_guide else 1.0 
                    for is_guide in place_reviews['isLocalGuide']]
            
            # Calculate weighted sentiment metrics
            avg_sentiment = self.calculate_weighted_average(place_reviews['sentiment_score'], weights)
            sentiment_std = np.sqrt(self.calculate_weighted_average((place_reviews['sentiment_score'] - avg_sentiment)**2, weights))
            
            # Sentiment categories
            sentiment_categories = place_reviews['sentiment_category'].value_counts(normalize=True).to_dict()
            positive_pct = sentiment_categories.get('positive', 0) * 100
            negative_pct = sentiment_categories.get('negative', 0) * 100
            neutral_pct = sentiment_categories.get('neutral', 0) * 100
            
            # Sentiment evolution over time (if enough data)
            sentiment_trend = 0
            if len(place_reviews) >= 10:
                place_reviews = place_reviews.sort_values('review_date')
                # Split into two halves and compare sentiment
                halfway = len(place_reviews) // 2
                first_half = place_reviews.iloc[:halfway]
                second_half = place_reviews.iloc[halfway:]
                sentiment_trend = second_half['sentiment_score'].mean() - first_half['sentiment_score'].mean()
            
            # Create sentiment metrics dictionary
            place_sentiment = {
                'placeId': place_id,
                'avg_sentiment': avg_sentiment,
                'sentiment_std': sentiment_std,
                'positive_review_pct': positive_pct,
                'negative_review_pct': negative_pct,
                'neutral_review_pct': neutral_pct,
                'sentiment_trend': sentiment_trend,
            }
            
            sentiment_metrics.append(place_sentiment)
            
        self.sentiment_df = pd.DataFrame(sentiment_metrics)
        
        # Merge with main metrics
        self.metrics_df = pd.merge(self.metrics_df, self.sentiment_df, on='placeId', how='left')
        print(f"Sentiment metrics calculated for {len(self.sentiment_df)} establishments")
        return self.sentiment_df
        
    def calculate_aspect_sentiment(self):
        """Calculate aspect-based sentiment for each establishment."""
        print("Calculating aspect-based sentiment...")
        
        # Function to check if an aspect is mentioned in a review
        def has_aspect(text, aspect_keywords):
            if pd.isna(text) or text == '':
                return False
            text = text.lower()
            return any(keyword in text for keyword in aspect_keywords)
        
        # Set the weight multiplier for local guides
        local_guide_weight = 1.2  # Local guides get 20% more weight

        # Calculate aspect sentiment for each review
        for aspect, keywords in self.aspects.items():
            aspect_col = f'has_{aspect}'
            sentiment_col = f'{aspect}_sentiment'
            
            # Check if review mentions this aspect
            self.combined_reviews[aspect_col] = self.combined_reviews['review_text'].apply(
                lambda x: has_aspect(x, keywords)
            )
            
            # Calculate sentiment for reviews mentioning this aspect
            self.combined_reviews[sentiment_col] = np.nan
            for idx, review in self.combined_reviews.iterrows():
                if review[aspect_col]:
                    # Use TextBlob for aspect sentiment (alternative approach)
                    text = str(review['review_text']).lower()
                    
                    # Find sentences containing aspect keywords
                    sentences = re.split(r'[.!?]', text)
                    aspect_sentences = [s for s in sentences if any(keyword in s for keyword in keywords)]
                    
                    if aspect_sentences:
                        aspect_sentiment = np.mean([TextBlob(s).sentiment.polarity for s in aspect_sentences])
                        self.combined_reviews.at[idx, sentiment_col] = aspect_sentiment
        
        # Calculate aspect sentiment metrics for each establishment
        aspect_metrics = []
        
        for place_id in self.metrics_df['placeId']:
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            if len(place_reviews) == 0:
                continue
                
            # For each aspect, calculate metrics
            aspect_data = {'placeId': place_id}
            
            for aspect in self.aspects.keys():
                aspect_col = f'has_{aspect}'
                sentiment_col = f'{aspect}_sentiment'
                
                # How often is this aspect mentioned
                mention_rate = place_reviews[aspect_col].mean() * 100
                aspect_data[f'{aspect}_mention_rate'] = mention_rate
                
                # Average sentiment for this aspect
                aspect_reviews = place_reviews[place_reviews[aspect_col] == True]
                if len(aspect_reviews) > 0:
                    # Apply weights based on local guide status
                    weights = [local_guide_weight if is_guide else 1.0 
                            for is_guide in aspect_reviews['isLocalGuide']]
                    avg_aspect_sentiment = self.calculate_weighted_average(
                        aspect_reviews[f'{aspect}_sentiment'], weights)
                    aspect_data[f'{aspect}_sentiment'] = avg_aspect_sentiment
                else:
                    aspect_data[f'{aspect}_sentiment'] = np.nan
            
            aspect_metrics.append(aspect_data)
        
        self.aspect_df = pd.DataFrame(aspect_metrics)
        
        # Merge with main metrics
        self.metrics_df = pd.merge(self.metrics_df, self.aspect_df, on='placeId', how='left')
        print(f"Aspect-based sentiment calculated for {len(self.aspect_df)} establishments")
        return self.aspect_df
    
    def perform_topic_modeling(self, n_topics=5, n_words=10, min_reviews=20):
        """
        Perform topic modeling on reviews for each establishment and return metrics for ranking.
        
        Parameters:
        -----------
        n_topics : int, optional
            Number of topics to extract (default: 5)
        n_words : int, optional
            Number of top words to extract for each topic (default: 10)
        min_reviews : int, optional
            Minimum number of reviews required to perform topic modeling (default: 20)
            
        Returns:
        --------
        dict
            Dictionary with topic modeling results indexed by placeId
        """
        print("Performing topic modeling...")
        
        # Initialize topic modeling results
        self.topic_results = {}
        
        # Create a count vectorizer with enhanced preprocessing
        stop_words = set(stopwords.words('english'))
        # Add domain-specific stop words
        domain_stop_words = ['hair', 'transplant', 'clinic', 'istanbul', 'turkey']
        stop_words.update(domain_stop_words)
        
        vectorizer = CountVectorizer(
            max_df=0.9,  # Ignore terms that appear in more than 90% of documents
            min_df=3,    # Ignore terms that appear in fewer than 3 documents
            stop_words=list(stop_words),
            max_features=1000,
            ngram_range=(1, 2)  # Include both unigrams and bigrams
        )
        
        # For each establishment with sufficient reviews, perform topic modeling
        for place_id in self.metrics_df['placeId']:
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            # Skip if not enough reviews
            if len(place_reviews) < min_reviews:
                continue
                
            # Prepare the documents with better preprocessing
            documents = []
            review_ids = []
            
            for idx, review in place_reviews.iterrows():
                if pd.isna(review['review_text']) or review['review_text'] == '':
                    continue
                    
                # Clean and normalize text
                text = review['review_text'].lower()
                text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
                text = re.sub(r'\s+', ' ', text).strip()  # Normalize whitespace
                
                if len(text.split()) > 3:  # Only include reviews with more than 3 words
                    documents.append(text)
                    review_ids.append(idx)
            
            if len(documents) < min_reviews:
                continue
                
            # Vectorize the text
            try:
                X = vectorizer.fit_transform(documents)
                
                # Get feature names
                feature_names = vectorizer.get_feature_names_out()
                
                # Create and fit LDA model with optimized parameters
                lda = LatentDirichletAllocation(
                    n_components=n_topics,
                    max_iter=20,
                    learning_method='online',
                    learning_offset=50.,
                    random_state=42,
                    doc_topic_prior=0.1,  # Smoother topic distribution
                    topic_word_prior=0.01  # Sparser word distribution
                )
                
                lda.fit(X)
                
                # Get topic distribution for each document
                topic_distribution = lda.transform(X)
                
                # Calculate coherence scores for each topic
                topic_coherence = []
                for topic_idx, topic in enumerate(lda.components_):
                    top_words_idx = topic.argsort()[:-n_words-1:-1]
                    
                    # Calculate simple coherence based on co-occurrence
                    coherence_score = 0
                    for i in range(len(top_words_idx)):
                        for j in range(i+1, len(top_words_idx)):
                            # Count documents with both words
                            word_i_docs = X[:, top_words_idx[i]].toarray().flatten() > 0
                            word_j_docs = X[:, top_words_idx[j]].toarray().flatten() > 0
                            both_words = np.logical_and(word_i_docs, word_j_docs).sum()
                            # Add log of co-occurrence probability
                            if both_words > 0:
                                coherence_score += np.log(both_words / len(documents))
                    
                    # Normalize by number of word pairs
                    n_pairs = len(top_words_idx) * (len(top_words_idx) - 1) / 2
                    if n_pairs > 0:
                        coherence_score /= n_pairs
                    
                    topic_coherence.append(coherence_score)
                
                # Extract top words for each topic
                topics = []
                for topic_idx, topic in enumerate(lda.components_):
                    top_words_idx = topic.argsort()[:-n_words-1:-1]
                    top_words = [feature_names[i] for i in top_words_idx]
                    
                    # Calculate document coverage for this topic
                    # (percentage of documents where this topic has the highest probability)
                    primary_topic_count = sum(np.argmax(topic_distribution, axis=1) == topic_idx)
                    document_coverage = primary_topic_count / len(topic_distribution) * 100
                    
                    topics.append({
                        'words': top_words,
                        'coherence': topic_coherence[topic_idx],
                        'document_coverage': document_coverage,
                        'word_weights': topic[top_words_idx].tolist()
                    })
                
                # Calculate topic diversity (entropy of topic distribution)
                # Higher entropy means more evenly distributed topics, indicating more diverse content
                topic_entropies = []
                for doc_topic_dist in topic_distribution:
                    # Normalize distribution to sum to 1
                    doc_topic_dist = doc_topic_dist / doc_topic_dist.sum()
                    # Calculate entropy
                    topic_entropy = -np.sum(doc_topic_dist * np.log2(doc_topic_dist + 1e-10))
                    # Normalize by max possible entropy (log2(n_topics))
                    topic_entropy /= np.log2(n_topics)
                    topic_entropies.append(topic_entropy)
                
                avg_topic_entropy = np.mean(topic_entropies)
                
                # Calculate topic clarity (inverse of perplexity)
                # Lower perplexity means the model better explains the data
                perplexity = lda.perplexity(X)
                topic_clarity = 1 / (1 + perplexity / 100)  # Normalize to 0-1 range
                
                # Calculate topic concentration (Gini coefficient of topic distribution)
                # Higher concentration means reviews focus on fewer topics
                topic_concentrations = []
                for doc_topic_dist in topic_distribution:
                    # Sort probabilities in ascending order
                    sorted_probs = np.sort(doc_topic_dist)
                    # Calculate Lorenz curve
                    cumulative_probs = np.cumsum(sorted_probs)
                    # Normalize to sum to 1
                    lorenz_curve = cumulative_probs / cumulative_probs[-1]
                    # Calculate Gini coefficient
                    # (area between line of equality and Lorenz curve) / (area under line of equality)
                    n = len(sorted_probs)
                    indices = np.arange(1, n + 1)
                    gini = 1 - 2 * np.sum((indices / n) - lorenz_curve) / n
                    topic_concentrations.append(gini)
                
                avg_topic_concentration = np.mean(topic_concentrations)
                
                # Calculate alignment between topics and sentiment
                if 'sentiment_score' in place_reviews.columns:
                    # Assign each review to its primary topic
                    primary_topics = np.argmax(topic_distribution, axis=1)
                    
                    # Calculate average sentiment for each topic
                    topic_sentiments = []
                    for i in range(n_topics):
                        topic_reviews = [review_ids[j] for j, topic in enumerate(primary_topics) if topic == i]
                        if topic_reviews:
                            avg_sentiment = place_reviews.loc[topic_reviews, 'sentiment_score'].mean()
                            topic_sentiments.append(avg_sentiment)
                        else:
                            topic_sentiments.append(0)
                    
                    # Calculate sentiment variance across topics
                    sentiment_variance = np.var(topic_sentiments)
                    
                    # Calculate sentiment concentration (are positive/negative sentiments clustered in specific topics?)
                    pos_topic_probs = np.zeros(n_topics)
                    neg_topic_probs = np.zeros(n_topics)
                    
                    for j, review_id in enumerate(review_ids):
                        sentiment = place_reviews.loc[review_id, 'sentiment_score']
                        if sentiment > 0.2:  # Positive sentiment
                            pos_topic_probs += topic_distribution[j]
                        elif sentiment < -0.2:  # Negative sentiment
                            neg_topic_probs += topic_distribution[j]
                    
                    # Normalize
                    if pos_topic_probs.sum() > 0:
                        pos_topic_probs /= pos_topic_probs.sum()
                    if neg_topic_probs.sum() > 0:
                        neg_topic_probs /= neg_topic_probs.sum()
                    
                    # Calculate Jensen-Shannon divergence between positive and negative topic distributions
                    # Higher divergence means positive and negative reviews focus on different topics
                    if pos_topic_probs.sum() > 0 and neg_topic_probs.sum() > 0:
                        m = 0.5 * (pos_topic_probs + neg_topic_probs)
                        sentiment_topic_divergence = 0.5 * (
                            entropy(pos_topic_probs, m) + entropy(neg_topic_probs, m)
                        )
                    else:
                        sentiment_topic_divergence = 0
                else:
                    topic_sentiments = [0] * n_topics
                    sentiment_variance = 0
                    sentiment_topic_divergence = 0
                
                # Store key metrics for ranking
                metrics = {
                    'topic_diversity': avg_topic_entropy,
                    'topic_coherence': np.mean(topic_coherence),
                    'topic_clarity': topic_clarity,
                    'topic_concentration': avg_topic_concentration,
                    'sentiment_variance': sentiment_variance,
                    'sentiment_topic_divergence': sentiment_topic_divergence,
                    'top_topic_coverage': max([topic['document_coverage'] for topic in topics]),
                    'review_coverage': len(documents) / len(place_reviews) * 100,  # % of reviews with meaningful content
                }
                
                # Store the results
                self.topic_results[place_id] = {
                    'model': lda,
                    'vectorizer': vectorizer,
                    'topics': topics,
                    'metrics': metrics,
                    'topic_sentiments': topic_sentiments
                }
                
                # Update metrics DataFrame with key metrics for ranking
                idx = self.metrics_df[self.metrics_df['placeId'] == place_id].index
                if len(idx) > 0:
                    for metric_name, metric_value in metrics.items():
                        self.metrics_df.at[idx[0], metric_name] = metric_value
            
            except Exception as e:
                print(f"Error performing topic modeling for establishment {place_id}: {str(e)}")
                continue
        
        print(f"Topic modeling completed for {len(self.topic_results)} establishments")
        return self.topic_results
    
    def analyze_word_choice(self):
        """Analyze distinctive word usage in positive and negative reviews."""
        print("Analyzing word choice patterns...")
        
        # Function to extract significant words
        def extract_significant_words(text_series, min_count=3):
            # Combine all text
            all_text = ' '.join(text_series.fillna('').astype(str))
            
            # Tokenize
            words = word_tokenize(all_text.lower())
            
            # Remove stopwords and punctuation
            stop_words = set(stopwords.words('english'))
            words = [word for word in words if word.isalpha() and word not in stop_words]
            
            # Count word frequencies
            word_freq = Counter(words)
            
            # Filter by minimum count
            return {word: count for word, count in word_freq.items() if count >= min_count}
        
        # For each establishment, analyze word choice
        word_metrics = []
        
        for place_id in self.metrics_df['placeId']:
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            if len(place_reviews) < 10:
                continue
                
            # Split reviews by sentiment
            positive_reviews = place_reviews[place_reviews['sentiment_category'] == 'positive']
            negative_reviews = place_reviews[place_reviews['sentiment_category'] == 'negative']
            
            # Extract significant words
            positive_words = extract_significant_words(positive_reviews['review_text'])
            negative_words = extract_significant_words(negative_reviews['review_text'])
            
            # Find distinctive words (higher frequency in one category)
            distinctive_positive = []
            distinctive_negative = []
            
            for word, pos_count in positive_words.items():
                neg_count = negative_words.get(word, 0)
                if pos_count > 2 * neg_count and pos_count >= 3:
                    distinctive_positive.append((word, pos_count))
            
            for word, neg_count in negative_words.items():
                pos_count = positive_words.get(word, 0)
                if neg_count > 2 * pos_count and neg_count >= 3:
                    distinctive_negative.append((word, neg_count))
            
            # Sort by frequency
            distinctive_positive.sort(key=lambda x: x[1], reverse=True)
            distinctive_negative.sort(key=lambda x: x[1], reverse=True)
            
            # Take top words
            top_positive = [word for word, _ in distinctive_positive[:10]]
            top_negative = [word for word, _ in distinctive_negative[:10]]
            
            # Calculate superlative usage
            superlatives = ['best', 'worst', 'amazing', 'terrible', 'excellent', 'awful', 
                            'outstanding', 'horrible', 'fantastic', 'abysmal']
            
            def count_superlatives(text):
                if pd.isna(text) or text == '':
                    return 0
                text = text.lower()
                return sum(1 for sup in superlatives if sup in text)
            
            superlative_counts = place_reviews['review_text'].apply(count_superlatives)
            superlative_rate = superlative_counts.sum() / len(place_reviews)
            
            # Create word metrics
            place_words = {
                'placeId': place_id,
                'distinctive_positive_words': ', '.join(top_positive),
                'distinctive_negative_words': ', '.join(top_negative),
                'superlative_usage_rate': superlative_rate
            }
            
            word_metrics.append(place_words)
        
        self.word_df = pd.DataFrame(word_metrics)
        
        # Merge with main metrics
        self.metrics_df = pd.merge(self.metrics_df, self.word_df, on='placeId', how='left')
        print(f"Word choice analysis completed for {len(self.word_df)} establishments")
        return self.word_df
    
    def analyze_named_entities(self):
        """Extract and analyze named entities mentioned in reviews."""
        print("Analyzing named entities...")
        
        # Initialize entity metrics
        entity_metrics = []
        
        for place_id in self.metrics_df['placeId']:
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            if len(place_reviews) < 5:
                continue
                
            # Combine all review text
            all_text = ' '.join(place_reviews['review_text'].fillna('').astype(str))
            
            if not all_text.strip():
                continue
                
            # Process with spaCy
            doc = nlp(all_text[:100000])  # Limit text length to avoid memory issues
            
            # Extract entities by type
            entities = {'PERSON': [], 'ORG': [], 'PRODUCT': [], 'GPE': [], 'LOC': [], 'FAC': []}
            
            for ent in doc.ents:
                if ent.label_ in entities:
                    entities[ent.label_].append(ent.text.lower())
            
            # Count entity frequencies
            entity_counts = {}
            for entity_type, entity_list in entities.items():
                entity_counts[entity_type] = Counter(entity_list)
            
            # Get top entities for each type
            top_entities = {}
            for entity_type, counts in entity_counts.items():
                top = [entity for entity, count in counts.most_common(5) if count >= 2]
                top_entities[f'top_{entity_type.lower()}'] = ', '.join(top)
            
            # Create entity metrics
            place_entities = {'placeId': place_id}
            place_entities.update(top_entities)
            
            # Calculate entity diversity (number of unique entities)
            entity_diversity = sum(len(counts) for counts in entity_counts.values())
            place_entities['entity_diversity'] = entity_diversity
            
            entity_metrics.append(place_entities)
        
        self.entity_df = pd.DataFrame(entity_metrics)
        
        # Merge with main metrics
        self.metrics_df = pd.merge(self.metrics_df, self.entity_df, on='placeId', how='left')
        print(f"Named entity analysis completed for {len(self.entity_df)} establishments")
        return self.entity_df
    
    def analyze_temporal_patterns(self):
        """Analyze temporal patterns in reviews."""
        print("Analyzing temporal patterns...")
        
        temporal_metrics = []
        
        for place_id in self.metrics_df['placeId']:
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            if len(place_reviews) < 10:
                continue
                
            # Sort reviews by date
            place_reviews = place_reviews.sort_values('review_date')
            
            # Group by month and calculate average rating and sentiment
            place_reviews['year_month'] = place_reviews['review_date'].dt.to_period('M')
            monthly_metrics = place_reviews.groupby('year_month').agg({
                'rating': 'mean',
                'sentiment_score': 'mean',
                'placeId': 'count'
            }).rename(columns={'placeId': 'review_count'})
            
            # Calculate rating trend (linear regression slope)
            if len(monthly_metrics) >= 3:
                x = np.arange(len(monthly_metrics))
                y = monthly_metrics['rating'].values
                
                if not np.isnan(y).all():
                    # Remove NaN values
                    valid_mask = ~np.isnan(y)
                    x_valid = x[valid_mask]
                    y_valid = y[valid_mask]
                    
                    if len(y_valid) >= 3:
                        # Calculate slope
                        A = np.vstack([x_valid, np.ones(len(x_valid))]).T
                        slope, _ = np.linalg.lstsq(A, y_valid, rcond=None)[0]
                        rating_trend = slope
                    else:
                        rating_trend = 0
                else:
                    rating_trend = 0
            else:
                rating_trend = 0
            
            # Calculate sentiment trend
            if len(monthly_metrics) >= 3:
                x = np.arange(len(monthly_metrics))
                y = monthly_metrics['sentiment_score'].values
                
                if not np.isnan(y).all():
                    # Remove NaN values
                    valid_mask = ~np.isnan(y)
                    x_valid = x[valid_mask]
                    y_valid = y[valid_mask]
                    
                    if len(y_valid) >= 3:
                        # Calculate slope
                        A = np.vstack([x_valid, np.ones(len(x_valid))]).T
                        slope, _ = np.linalg.lstsq(A, y_valid, rcond=None)[0]
                        sentiment_trend = slope
                    else:
                        sentiment_trend = 0
                else:
                    sentiment_trend = 0
            else:
                sentiment_trend = 0
            
            # Check for seasonality (if enough data)
            has_seasonality = False
            seasonal_pattern = ''
            
            if len(monthly_metrics) >= 12:
                # Group by month and calculate average metrics
                place_reviews['month'] = place_reviews['review_date'].dt.month
                monthly_avg = place_reviews.groupby('month').agg({
                    'rating': 'mean',
                    'sentiment_score': 'mean',
                    'placeId': 'count'
                }).rename(columns={'placeId': 'review_count'})
                
                # Check if certain months consistently have higher/lower ratings
                month_rating_z = (monthly_avg['rating'] - monthly_avg['rating'].mean()) / monthly_avg['rating'].std()
                month_sentiment_z = (monthly_avg['sentiment_score'] - monthly_avg['sentiment_score'].mean()) / monthly_avg['sentiment_score'].std()
                
                # Find months with significant deviations
                high_months = [str(m) for m in monthly_avg.index[month_rating_z > 1]]
                low_months = [str(m) for m in monthly_avg.index[month_rating_z < -1]]
                
                if high_months or low_months:
                    has_seasonality = True
                    seasonal_pattern = f"Higher ratings in months: {', '.join(high_months)}. " if high_months else ""
                    seasonal_pattern += f"Lower ratings in months: {', '.join(low_months)}." if low_months else ""
            
            # Create temporal metrics
            place_temporal = {
                'placeId': place_id,
                'rating_trend': rating_trend,
                'sentiment_trend': sentiment_trend,
                'has_seasonality': has_seasonality,
                'seasonal_pattern': seasonal_pattern
            }
            
            temporal_metrics.append(place_temporal)
        
        self.temporal_df = pd.DataFrame(temporal_metrics)
        
        # Merge with main metrics
        self.metrics_df = pd.merge(self.metrics_df, self.temporal_df, on='placeId', how='left')
        print(f"Temporal analysis completed for {len(self.temporal_df)} establishments")
        return self.temporal_df
    
    def detect_review_authenticity(self):
        """Detect potential signals of review authenticity issues."""
        print("Analyzing review authenticity signals...")
        
        authenticity_metrics = []
        
        for place_id in self.metrics_df['placeId']:
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            if len(place_reviews) < 10:
                continue
                
            # Check for unusual rating patterns
            rating_counts = place_reviews['rating'].value_counts(normalize=True)
            
            # Most reviews are 5-star or 1-star (polarized distribution)
            polarization = (rating_counts.get(5, 0) + rating_counts.get(1, 0))
            is_polarized = polarization > 0.8
            
            # Unusually high proportion of 5-star ratings
            high_five_star = rating_counts.get(5, 0) > 0.9
            
            # Check for clustering of review timestamps
            place_reviews = place_reviews.sort_values('review_date')
            if len(place_reviews) >= 5:
                # Calculate time differences between consecutive reviews
                place_reviews['time_diff'] = place_reviews['review_date'].diff().dt.total_seconds() / 3600  # hours
                
                # Check for clusters (many reviews posted within a short timeframe)
                clusters = []
                current_cluster = []
                
                for idx, row in place_reviews.iterrows():
                    if pd.isna(row['time_diff']) or row['time_diff'] > 24:  # New cluster if >24 hours gap
                        if len(current_cluster) >= 3:  # Consider it a cluster if 3+ reviews
                            clusters.append(current_cluster)
                        current_cluster = [idx]
                    else:
                        current_cluster.append(idx)
                
                # Add the last cluster if it's significant
                if len(current_cluster) >= 3:
                    clusters.append(current_cluster)
                
                has_review_clusters = len(clusters) > 0
                cluster_count = len(clusters)
                largest_cluster = max([len(c) for c in clusters]) if clusters else 0
            else:
                has_review_clusters = False
                cluster_count = 0
                largest_cluster = 0
            
            # Check for review similarity among clusters using TF-IDF within clusters
            cluster_similarities = []
            
            # Only perform similarity analysis if clusters were detected
            if has_review_clusters and clusters:
                for cluster_idx, cluster in enumerate(clusters):
                    # If cluster has enough reviews to perform meaningful similarity analysis
                    if len(cluster) >= 3:
                        cluster_reviews = place_reviews.loc[cluster, 'review_text'].fillna('')
                        
                        # Use TF-IDF to measure review similarity within this specific cluster
                        tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
                        try:
                            # Create TF-IDF matrix just for this cluster
                            cluster_tfidf_matrix = tfidf_vectorizer.fit_transform(cluster_reviews)
                            
                            # Calculate pairwise cosine similarity within the cluster
                            from sklearn.metrics.pairwise import cosine_similarity
                            cluster_similarity_matrix = cosine_similarity(cluster_tfidf_matrix)
                            
                            # Remove self-similarity (diagonal)
                            np.fill_diagonal(cluster_similarity_matrix, 0)
                            
                            # Calculate average similarity within this cluster
                            cluster_avg_similarity = cluster_similarity_matrix.sum() / (len(cluster) * (len(cluster) - 1))
                            
                            # Track similarity for each cluster
                            cluster_similarities.append({
                                'cluster_idx': cluster_idx,
                                'size': len(cluster),
                                'avg_similarity': cluster_avg_similarity
                            })
                            
                        except Exception as e:
                            print(f"Error calculating similarity for cluster {cluster_idx} of {place_id}: {e}")
                
                # Calculate overall metrics from cluster similarities
                if cluster_similarities:
                    # Average similarity across all clusters
                    avg_cluster_similarity = np.mean([cs['avg_similarity'] for cs in cluster_similarities])
                    
                    # Maximum cluster similarity (most suspicious cluster)
                    max_cluster_similarity = np.max([cs['avg_similarity'] for cs in cluster_similarities])
                    
                    # Number of suspicious clusters (with avg similarity > 0.5)
                    suspicious_clusters = sum(1 for cs in cluster_similarities if cs['avg_similarity'] > 0.5)

                else:
                    avg_cluster_similarity = 0
                    max_cluster_similarity = 0
                    suspicious_clusters = 0
            else:
                avg_cluster_similarity = 0
                max_cluster_similarity = 0
                suspicious_clusters = 0
            
            # General similarity check for all reviews
            similar_review_rate = 0
            if len(place_reviews) >= 5:
                # Use TF-IDF to measure review similarity across all reviews
                tfidf_vectorizer = TfidfVectorizer(min_df=1, stop_words='english')
                try:
                    tfidf_matrix = tfidf_vectorizer.fit_transform(place_reviews['review_text'].fillna(''))
                    
                    # Calculate pairwise cosine similarity
                    from sklearn.metrics.pairwise import cosine_similarity
                    similarity_matrix = cosine_similarity(tfidf_matrix)
                    
                    # Remove self-similarity (diagonal)
                    np.fill_diagonal(similarity_matrix, 0)
                    
                    # Find highly similar review pairs (similarity > 0.7)
                    similar_pairs = 0
                    for i in range(len(similarity_matrix)):
                        for j in range(i+1, len(similarity_matrix)):
                            if similarity_matrix[i, j] > 0.7:
                                similar_pairs += 1
                    
                    similar_review_rate = similar_pairs / (len(place_reviews) * (len(place_reviews) - 1) / 2)
                except Exception as e:
                    print(f"Error calculating general similarity: {e}")
                    similar_review_rate = 0
            
            # Create authenticity metrics
            place_authenticity = {
                'placeId': place_id,
                'has_review_clusters': has_review_clusters,
                'cluster_count': cluster_count,
                'largest_cluster_size': largest_cluster,
                'avg_cluster_similarity': avg_cluster_similarity,
                'max_cluster_similarity': max_cluster_similarity,
                'suspicious_clusters': suspicious_clusters,
                'similar_review_rate': similar_review_rate,
                'authenticity_concerns': (has_review_clusters and max_cluster_similarity > 0.5) or similar_review_rate > 0.1
            }

            authenticity_metrics.append(place_authenticity)
        
        self.authenticity_df = pd.DataFrame(authenticity_metrics)
        
        # Merge with main metrics
        self.metrics_df = pd.merge(self.metrics_df, self.authenticity_df, on='placeId', how='left')
        print(f"Review authenticity analysis completed for {len(self.authenticity_df)} establishments")
        return self.authenticity_df
    
    def analyze_similarity_outliers(self):
        """
        Analyze establishments with abnormally high similarity in review clusters
        using the Interquartile Range (IQR) method.
        """
        print("Analyzing review cluster similarity outliers...")
        
        # Skip if authenticity metrics aren't available
        if 'avg_cluster_similarity' not in self.metrics_df.columns:
            print("ERROR: avg_cluster_similarity metric not found. Run detect_review_authenticity first.")
            return None
        
        # Get all non-null avg_cluster_similarity values
        similarity_values = self.metrics_df['avg_cluster_similarity'].dropna()
        
        if len(similarity_values) < 5:
            print("Not enough data to calculate similarity outliers.")
            return None
        
        # Calculate IQR statistics
        Q1 = similarity_values.quantile(0.25)
        Q3 = similarity_values.quantile(0.75)
        IQR = Q3 - Q1
        
        # Define outlier thresholds
        mild_threshold = Q3 + (1.5 * IQR)
        extreme_threshold = Q3 + (3 * IQR)
        
        # Classify establishments
        self.metrics_df['similarity_comparison'] = 'normal'
        
        # Identify mild outliers
        mild_mask = (self.metrics_df['avg_cluster_similarity'] > mild_threshold) & \
                (self.metrics_df['avg_cluster_similarity'] <= extreme_threshold)
        self.metrics_df.loc[mild_mask, 'similarity_comparison'] = 'suspicious'
        
        # Identify extreme outliers
        extreme_mask = self.metrics_df['avg_cluster_similarity'] > extreme_threshold
        self.metrics_df.loc[extreme_mask, 'similarity_comparison'] = 'highly suspicious'
        
        # Create percentile rank for each establishment
        self.metrics_df['similarity_percentile'] = self.metrics_df['avg_cluster_similarity'].rank(pct=True) * 100
        
        # Calculate how many standard deviations from the mean
        mean_similarity = similarity_values.mean()
        std_similarity = similarity_values.std()
        
        if std_similarity > 0:  # Avoid division by zero
            self.metrics_df['similarity_z_score'] = (self.metrics_df['avg_cluster_similarity'] - mean_similarity) / std_similarity
        else:
            self.metrics_df['similarity_z_score'] = 0
        
        # Count outliers
        mild_outliers = sum(mild_mask)
        extreme_outliers = sum(extreme_mask)
        
        print(f"Found {mild_outliers} establishments with suspicious review similarity")
        print(f"Found {extreme_outliers} establishments with highly suspicious review similarity")
        
        # Ensure the metric is used in the detect_review_authenticity method
        if hasattr(self, 'authenticity_df') and not self.authenticity_df.empty:
            # Update authenticity metrics
            for idx, row in self.metrics_df.iterrows():
                if row['similarity_comparison'] in ['suspicious', 'highly suspicious']:
                    place_id = row['placeId']
                    
                    # Set authenticity_concerns to True for suspicious establishments
                    auth_idx = self.authenticity_df[self.authenticity_df['placeId'] == place_id].index
                    if len(auth_idx) > 0:
                        self.authenticity_df.at[auth_idx[0], 'authenticity_concerns'] = True
                        self.metrics_df.at[idx, 'authenticity_concerns'] = True
                        
                        # Add a new alert column specific to similarity concerns
                        self.authenticity_df.at[auth_idx[0], 'cluster_similarity_alert'] = True
                        self.metrics_df.at[idx, 'cluster_similarity_alert'] = True
                    else:
                        self.authenticity_df.at[auth_idx[0], 'cluster_similarity_alert'] = False
                        self.metrics_df.at[idx, 'cluster_similarity_alert'] = False

        
        return self.metrics_df[['placeId', 'title', 'avg_cluster_similarity', 
                            'similarity_comparison', 'similarity_percentile', 'similarity_z_score']]

    def analyze_comparative_references(self):
        """Extract and analyze comparative references in reviews."""
        print("Analyzing comparative references...")
        
        comparative_metrics = []
        
        # Define comparison keywords
        comparison_terms = [
            'better than', 'worse than', 'best', 'worst', 'compared to',
            'superior to', 'inferior to', 'beats', 'exceeds', 'surpasses',
            'falls short', 'not as good as', 'prefer', 'preferable', 
            'compared with', 'in comparison', 'unlike', 'similar to'
        ]
        
        for place_id in self.metrics_df['placeId']:
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            if len(place_reviews) < 5:
                continue
                
            # Identify reviews with comparative language
            comparison_counts = {'better': 0, 'worse': 0, 'neutral': 0}
            comparative_examples = []
            
            for idx, review in place_reviews.iterrows():
                if pd.isna(review['review_text']) or review['review_text'] == '':
                    continue
                    
                text = review['review_text'].lower()
                
                # Check if any comparison term exists
                has_comparison = any(term in text for term in comparison_terms)
                
                if has_comparison:
                    # Identify comparison direction
                    positive_comparison = any(term in text for term in ['better than', 'best', 'superior to', 'exceeds', 'surpasses', 'prefer'])
                    negative_comparison = any(term in text for term in ['worse than', 'worst', 'inferior to', 'falls short', 'not as good as'])
                    
                    if positive_comparison and not negative_comparison:
                        comparison_counts['better'] += 1
                    elif negative_comparison and not positive_comparison:
                        comparison_counts['worse'] += 1
                    else:
                        comparison_counts['neutral'] += 1
                    
                    # Extract comparison sentence
                    sentences = re.split(r'[.!?]', text)
                    comparative_sentences = [s.strip() for s in sentences if any(term in s for term in comparison_terms)]
                    
                    if comparative_sentences:
                        sentiment = 'positive' if positive_comparison else ('negative' if negative_comparison else 'neutral')
                        comparative_examples.append((comparative_sentences[0], sentiment))
            
            # Calculate comparison metrics
            total_comparisons = sum(comparison_counts.values())
            comparison_rate = total_comparisons / len(place_reviews)
            
            positive_comparison_rate = comparison_counts['better'] / total_comparisons if total_comparisons > 0 else 0
            negative_comparison_rate = comparison_counts['worse'] / total_comparisons if total_comparisons > 0 else 0
            
            # Create comparative metrics
            place_comparative = {
                'placeId': place_id,
                'comparison_rate': comparison_rate,
                'positive_comparison_rate': positive_comparison_rate,
                'negative_comparison_rate': negative_comparison_rate,
                'total_comparisons': total_comparisons
            }
            
            # Add example comparative statements
            if comparative_examples:
                positive_examples = [ex for ex, sentiment in comparative_examples if sentiment == 'positive']
                negative_examples = [ex for ex, sentiment in comparative_examples if sentiment == 'negative']
                
                place_comparative['positive_comparison_example'] = positive_examples[0] if positive_examples else ''
                place_comparative['negative_comparison_example'] = negative_examples[0] if negative_examples else ''
            
            comparative_metrics.append(place_comparative)
        
        self.comparative_df = pd.DataFrame(comparative_metrics)
        
        # Merge with main metrics
        self.metrics_df = pd.merge(self.metrics_df, self.comparative_df, on='placeId', how='left')
        print(f"Comparative reference analysis completed for {len(self.comparative_df)} establishments")
        return self.comparative_df
    
    def analyze_customer_journey(self):
        """Analyze customer journey markers in reviews."""
        print("Analyzing customer journey markers...")
        
        journey_metrics = []
        
        # Define journey marker keywords
        first_time_markers = [
            'first time', 'first visit', 'never been', 'tried for the first time',
            'new customer', 'new to this', 'never tried'
        ]
        
        repeat_markers = [
            'regular', 'again', 'return', 'back', 'repeat', 'loyal',
            'always go', 'always come', 'many times', 'several times',
            'frequent', 'weekly', 'monthly', 'year after year'
        ]
        
        will_return_markers = [
            'will return', 'will be back', 'coming back', 'will go again',
            'definitely return', 'visit again', 'return customer'
        ]
        
        wont_return_markers = [
            'never return', 'never going back', 'won\'t be back', 'won\'t return',
            'last time', 'never again', 'stay away', 'avoid this place'
        ]
        
        for place_id in self.metrics_df['placeId']:
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            if len(place_reviews) < 5:
                continue
                
            # Count journey markers
            first_time_count = 0
            repeat_count = 0
            will_return_count = 0
            wont_return_count = 0
            
            for idx, review in place_reviews.iterrows():
                if pd.isna(review['review_text']) or review['review_text'] == '':
                    continue
                    
                text = review['review_text'].lower()
                
                # Check for journey markers
                if any(marker in text for marker in first_time_markers):
                    first_time_count += 1
                    
                if any(marker in text for marker in repeat_markers):
                    repeat_count += 1
                    
                if any(marker in text for marker in will_return_markers):
                    will_return_count += 1
                    
                if any(marker in text for marker in wont_return_markers):
                    wont_return_count += 1
            
            # Calculate journey metrics
            total_reviews = len(place_reviews)
            first_time_rate = first_time_count / total_reviews
            repeat_rate = repeat_count / total_reviews
            will_return_rate = will_return_count / total_reviews
            wont_return_rate = wont_return_count / total_reviews
            
            # Customer loyalty score (repeat + will return - won't return)
            loyalty_score = (repeat_rate + will_return_rate - wont_return_rate) * 100
            
            # Create journey metrics
            place_journey = {
                'placeId': place_id,
                'first_time_rate': first_time_rate * 100,
                'repeat_customer_rate': repeat_rate * 100,
                'will_return_rate': will_return_rate * 100,
                'wont_return_rate': wont_return_rate * 100,
                'customer_loyalty_score': loyalty_score
            }
            
            journey_metrics.append(place_journey)
        
        self.journey_df = pd.DataFrame(journey_metrics)
        
        # Merge with main metrics
        self.metrics_df = pd.merge(self.metrics_df, self.journey_df, on='placeId', how='left')
        print(f"Customer journey analysis completed for {len(self.journey_df)} establishments")
        return self.journey_df
    
    def analyze_complaint_resolution(self):
        """Analyze complaint resolution in reviews and responses."""
        print("Analyzing complaint resolution patterns...")
        
        resolution_metrics = []
        
        # Define complaint and resolution keywords
        complaint_keywords = [
            'problem', 'issue', 'complaint', 'disappointing', 'disappointed',
            'frustrated', 'terrible', 'awful', 'bad', 'wrong', 'mistake',
            'error', 'fail', 'poor', 'unacceptable', 'dissatisfied'
        ]
        
        resolution_keywords = [
            'resolve', 'resolved', 'solution', 'solved', 'fix', 'fixed',
            'addressed', 'corrected', 'improved', 'apologize', 'sorry',
            'refund', 'replacement', 'compensate', 'compensation'
        ]
        
        for place_id in self.metrics_df['placeId']:
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            if len(place_reviews) < 5:
                continue
                
            # Identify reviews with complaints
            complaint_reviews = []
            
            for idx, review in place_reviews.iterrows():
                if pd.isna(review['review_text']) or review['review_text'] == '':
                    continue
                    
                text = review['review_text'].lower()
                
                # Check if review contains complaint keywords
                has_complaint = any(keyword in text for keyword in complaint_keywords)
                
                if has_complaint:
                    has_response = not pd.isna(review['response_text']) and review['response_text'] != ''
                    
                    if has_response:
                        response_text = review['response_text'].lower()
                        has_resolution_language = any(keyword in response_text for keyword in resolution_keywords)
                    else:
                        has_resolution_language = False
                        
                    complaint_reviews.append({
                        'review_id': idx,
                        'rating': review['rating'],
                        'has_response': has_response,
                        'response_time_days': review['response_time_days'],
                        'has_resolution_language': has_resolution_language
                    })
            
            # Calculate complaint resolution metrics
            total_complaints = len(complaint_reviews)
            
            if total_complaints > 0:
                complaint_df = pd.DataFrame(complaint_reviews)
                
                response_rate = complaint_df['has_response'].mean() * 100
                resolution_language_rate = complaint_df[complaint_df['has_response']]['has_resolution_language'].mean() * 100 if len(complaint_df[complaint_df['has_response']]) > 0 else 0
                avg_response_time = complaint_df['response_time_days'].mean()
                
                # Create resolution metrics
                place_resolution = {
                    'placeId': place_id,
                    'complaint_count': total_complaints,
                    'complaint_rate': total_complaints / len(place_reviews) * 100,
                    'complaint_response_rate': response_rate,
                    'resolution_language_rate': resolution_language_rate,
                    'complaint_response_time': avg_response_time
                }
            else:
                place_resolution = {
                    'placeId': place_id,
                    'complaint_count': 0,
                    'complaint_rate': 0,
                    'complaint_response_rate': np.nan,
                    'resolution_language_rate': np.nan,
                    'complaint_response_time': np.nan
                }
            
            resolution_metrics.append(place_resolution)
        
        self.resolution_df = pd.DataFrame(resolution_metrics)
        
        # Merge with main metrics
        self.metrics_df = pd.merge(self.metrics_df, self.resolution_df, on='placeId', how='left')
        print(f"Complaint resolution analysis completed for {len(self.resolution_df)} establishments")
        return self.resolution_df
    
    def create_composite_score(self):
        """Create a composite score for ranking establishments."""
        print("Creating composite scores and rankings...")
        
        # Ensure all necessary metrics are calculated
        if self.metrics_df is None or len(self.metrics_df) == 0:
            print("ERROR: No metrics available. Run analysis functions first.")
            return None
        
        # Define weights for different metric categories
        self.composite_score_weights = self.config["composite_score_weights"]
                
        # Create a copy of the metrics dataframe for scoring
        score_df = self.metrics_df.copy()
        
        # Normalize numerical features to 0-1 scale
        for column in self.composite_score_weights.keys():
            if column in score_df.columns and column != 'authenticity_concerns':
                # Skip non-numeric columns
                if not np.issubdtype(score_df[column].dtype, np.number):
                    continue
                    
                # Handle missing values
                score_df[column].fillna(score_df[column].mean(), inplace=True)
                
                # Min-max scaling
                min_val = score_df[column].min()
                max_val = score_df[column].max()
                
                if max_val > min_val:
                    score_df[column + '_normalized'] = (score_df[column] - min_val) / (max_val - min_val)
                else:
                    score_df[column + '_normalized'] = 0.5  # Default if all values are the same
        
        # Handle authenticity_concerns (boolean to numeric)
        if 'authenticity_concerns' in score_df.columns:
            score_df['authenticity_concerns_normalized'] = score_df['authenticity_concerns'].astype(float)
        
        # Calculate weighted score
        score_df['composite_score'] = 0
        
        for column, weight in self.composite_score_weights.items():
            normalized_col = column + '_normalized'
            if normalized_col in score_df.columns:
                score_df['composite_score'] += score_df[normalized_col] * weight
        
        # Scale final score to 0-100
        min_score = score_df['composite_score'].min()
        max_score = score_df['composite_score'].max()
        
        if max_score > min_score:
            score_df['final_score'] = ((score_df['composite_score'] - min_score) / (max_score - min_score)) * 100
        else:
            score_df['final_score'] = 50  # Default if all scores are the same

        # Replace NaN values in final_score with 0 before ranking
        score_df['final_score'].fillna(0, inplace=True)

        # Round scores to 2 decimal places
        score_df['final_score'] = score_df['final_score'].round(2)
        
        # Add rank
        score_df['rank'] = score_df['final_score'].rank(ascending=False, method='min').astype(int)
        
        # Extract final results
        self.results_df = score_df[['placeId', 'title', 'final_score', 'rank', 'avg_rating', 
                                    'total_reviews', 'avg_sentiment', 'customer_loyalty_score']]
        
        print(f"Rankings created for {len(self.results_df)} establishments")
        return self.results_df
    
    def generate_insights(self, top_n=10):
        """Generate insights for top-ranked establishments."""
        print("Generating insights for top establishments...")
        
        if self.results_df is None or len(self.results_df) == 0:
            print("ERROR: No results available. Run create_composite_score first.")
            return None
        
        # Get top N establishments
        top_establishments = self.results_df.sort_values('rank').head(top_n)
        
        insights = {}
        
        for _, establishment in top_establishments.iterrows():
            place_id = establishment['placeId']
            
            # Get full metrics for this establishment
            place_metrics = self.metrics_df[self.metrics_df['placeId'] == place_id].iloc[0]
            
            # Extract key strengths
            strengths = []
            
            # Rating and sentiment
            if hasattr(place_metrics, 'avg_rating') and place_metrics['avg_rating'] >= 4.5:
                strengths.append(f"Exceptional average rating of {place_metrics['avg_rating']:.1f}/5")
            
            if hasattr(place_metrics, 'positive_review_pct') and place_metrics['positive_review_pct'] >= 85:
                strengths.append(f"Very high positive sentiment ({place_metrics['positive_review_pct']:.1f}%)")
            
            # Customer loyalty
            if hasattr(place_metrics, 'customer_loyalty_score') and place_metrics['customer_loyalty_score'] >= 50:
                strengths.append(f"Strong customer loyalty ({place_metrics['customer_loyalty_score']:.1f})")
            
            if hasattr(place_metrics, 'repeat_customer_rate') and place_metrics['repeat_customer_rate'] >= 25:
                strengths.append(f"High repeat customer rate ({place_metrics['repeat_customer_rate']:.1f}%)")
            
            # Aspect sentiments
            for aspect in ['service', 'quality', 'ambiance', 'price']:
                aspect_col = f'{aspect}_sentiment'
                mention_col = f'{aspect}_mention_rate'
                
                if hasattr(place_metrics, aspect_col) and hasattr(place_metrics, mention_col):
                    if not pd.isna(place_metrics[aspect_col]) and place_metrics[aspect_col] >= 0.5 and place_metrics[mention_col] >= 10:
                        strengths.append(f"Highly rated {aspect}")
            
            # Extract areas for improvement
            improvements = []
            
            # Low ratings in specific aspects
            for aspect in ['service', 'quality', 'ambiance', 'price', 'cleanliness']:
                aspect_col = f'{aspect}_sentiment'
                mention_col = f'{aspect}_mention_rate'
                
                if hasattr(place_metrics, aspect_col) and hasattr(place_metrics, mention_col):
                    if not pd.isna(place_metrics[aspect_col]) and place_metrics[aspect_col] <= 0 and place_metrics[mention_col] >= 10:
                        improvements.append(f"Improve {aspect}")
            
            # Response rate
            if hasattr(place_metrics, 'response_rate') and place_metrics['response_rate'] < 50:
                improvements.append("Increase response rate to reviews")
            
            # Response time
            if hasattr(place_metrics, 'avg_response_time') and not pd.isna(place_metrics['avg_response_time']) and place_metrics['avg_response_time'] > 7:
                improvements.append("Reduce response time to reviews")
            
            # Negative trends
            if hasattr(place_metrics, 'rating_trend') and place_metrics['rating_trend'] < -0.02:
                improvements.append("Address declining ratings trend")
            
            # Authenticity concerns
            if hasattr(place_metrics, 'authenticity_concerns') and place_metrics['authenticity_concerns']:
                improvements.append("Address potential review authenticity issues")
            
            # Store insights
            insights[place_id] = {
                'name': establishment['title'],
                'score': establishment['final_score'],
                'rank': establishment['rank'],
                'strengths': strengths[:5],  # Top 5 strengths
                'improvements': improvements[:3]  # Top 3 improvements
            }
            
            # Add distinctive words if available
            if hasattr(place_metrics, 'distinctive_positive_words'):
                insights[place_id]['distinctive_positive'] = place_metrics['distinctive_positive_words']
                
            if hasattr(place_metrics, 'distinctive_negative_words'):
                insights[place_id]['distinctive_negative'] = place_metrics['distinctive_negative_words']
        
        self.insights = insights
        return insights
    
    def run_full_analysis(self):
        """Run the full analysis pipeline."""
        self.calculate_basic_metrics()
        self.calculate_sentiment_metrics()
        self.calculate_aspect_sentiment()
        self.perform_topic_modeling()
        self.analyze_word_choice()
        self.analyze_named_entities()
        self.analyze_temporal_patterns()
        self.detect_review_authenticity()
        self.analyze_similarity_outliers()
        self.analyze_comparative_references()
        self.analyze_customer_journey()
        self.analyze_complaint_resolution()
        self.create_composite_score()
        self.generate_insights()
        
        return self.results_df
    
    def save_results(self, output_dir='.'):
        """Save analysis results to Excel files."""
        print("Saving results...")

        # Get current date and time for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results
        if self.results_df is not None:
            self.results_df.to_excel(f"{output_dir}/establishment_rankings_{timestamp}.xlsx", index=False)
            print(f"Rankings saved to {output_dir}/establishment_rankings_{timestamp}.xlsx")
        
        # Save full metrics
        if self.metrics_df is not None:
            self.metrics_df.to_excel(f"{output_dir}/establishment_metrics_{timestamp}.xlsx", index=False)
            print(f"Full metrics saved to {output_dir}/establishment_metrics_{timestamp}.xlsx")
        
        # Save insights as JSON
        if self.insights is not None:
            import json
            with open(f"{output_dir}/establishment_insights_{timestamp}.json", 'w') as f:
                json.dump(self.insights, f, indent=2)
            print(f"Insights saved to {output_dir}/establishment_insights_{timestamp}.json")
        
        print("All results saved successfully")
    
    def save_results(self, output_dir='.'):
        """Save analysis results to Excel files."""
        print("Saving results...")

        # Get current date and time for file naming
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results with scoring components
        if self.results_df is not None:
            # Get the columns used in scoring
            scoring_columns = self.composite_score_weights.keys()
            
            # Create enhanced results with scoring fields
            enhanced_results = self.results_df.copy()
            
            # Add each scoring column from metrics_df to the results
            for col in scoring_columns:
                if col in self.metrics_df.columns:
                    enhanced_results[col] = self.metrics_df.set_index('placeId').loc[enhanced_results['placeId'], col].values
            
            # Add normalized values for better understanding
            for col in scoring_columns:
                norm_col = f"{col}_normalized"
                if norm_col in self.metrics_df.columns:
                    enhanced_results[f"{col}_normalized"] = self.metrics_df.set_index('placeId').loc[enhanced_results['placeId'], norm_col].values
            
            # Save to Excel
            enhanced_results.to_excel(f"{output_dir}/establishment_rankings_{timestamp}.xlsx", index=False)
            print(f"Enhanced rankings with scoring details saved to {output_dir}/establishment_rankings_{timestamp}.xlsx")
        
        # Save full metrics
        if self.metrics_df is not None:
            self.metrics_df.to_excel(f"{output_dir}/establishment_metrics_{timestamp}.xlsx", index=False)
            print(f"Full metrics saved to {output_dir}/establishment_metrics_{timestamp}.xlsx")
        
        # Save insights as JSON
        if self.insights is not None:
            import json
            with open(f"{output_dir}/establishment_insights_{timestamp}.json", 'w') as f:
                json.dump(self.insights, f, indent=2)
            print(f"Insights saved to {output_dir}/establishment_insights_{timestamp}.json")
        
        print("All results saved successfully")


# Example usage
if __name__ == "__main__":
    # File paths
    google_maps_file = "google_maps_example.csv"
    trustpilot_file = "trustpilot_example.csv"
    establishment_file = "establishment_base_example.csv"
    
    # Create analyzer
    analyzer = ReviewAnalyzer(google_maps_file, trustpilot_file, establishment_file)
    
    # Run full analysis
    results = analyzer.run_full_analysis()
    
    # Save results
    analyzer.save_results("results")
    
    # Print top 10 establishments
    print("\nTop 10 Establishments:")
    top10 = results.sort_values('rank').head(10)
    print(top10[['rank', 'title', 'final_score', 'avg_rating', 'total_reviews']])