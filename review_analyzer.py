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

# Download necessary NLTK resources
nltk.download('vader_lexicon')
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

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
            90: 0.9,    # Last quarter: 90%
            180: 0.8,   # Last 6 months: 80%
            365: 0.7,   # Last year: 70%
            730: 0.5,   # Last 2 years: 50%
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

    def calculate_basic_metrics(self):
        """Calculate basic metrics for each establishment."""
        print("Calculating basic metrics...")
        
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
            
            # Rating metrics
            avg_rating = place_reviews['rating'].mean()
            google_avg_rating = google_reviews['rating'].mean() if len(google_reviews) > 0 else np.nan
            trustpilot_avg_rating = trustpilot_reviews['rating'].mean() if len(trustpilot_reviews) > 0 else np.nan
            rating_std = place_reviews['rating'].std() if len(place_reviews) > 1 else 0
            
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
        """Calculate sentiment metrics for each establishment."""
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
        
        # Calculate sentiment metrics for each establishment
        sentiment_metrics = []
        
        for place_id in self.metrics_df['placeId']:
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            if len(place_reviews) == 0:
                continue
                
            # Overall sentiment metrics
            avg_sentiment = place_reviews['sentiment_score'].mean()
            sentiment_std = place_reviews['sentiment_score'].std() if len(place_reviews) > 1 else 0
            
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
                    avg_aspect_sentiment = aspect_reviews[sentiment_col].mean()
                    aspect_data[f'{aspect}_sentiment'] = avg_aspect_sentiment
                else:
                    aspect_data[f'{aspect}_sentiment'] = np.nan
            
            aspect_metrics.append(aspect_data)
        
        self.aspect_df = pd.DataFrame(aspect_metrics)
        
        # Merge with main metrics
        self.metrics_df = pd.merge(self.metrics_df, self.aspect_df, on='placeId', how='left')
        print(f"Aspect-based sentiment calculated for {len(self.aspect_df)} establishments")
        return self.aspect_df
    
    def perform_topic_modeling(self, n_topics=5, n_words=10):
        """Perform topic modeling on reviews for each establishment."""
        print("Performing topic modeling...")
        
        # Initialize topic modeling results
        self.topic_results = {}
        
        # Create a count vectorizer
        vectorizer = CountVectorizer(
            max_df=0.95, 
            min_df=2,
            stop_words='english',
            max_features=1000
        )
        
        # For each establishment with sufficient reviews, perform topic modeling
        for place_id in self.metrics_df['placeId']:
            place_reviews = self.combined_reviews[self.combined_reviews['placeId'] == place_id]
            
            # Skip if not enough reviews
            if len(place_reviews) < 20:
                continue
                
            # Prepare the documents
            documents = place_reviews['review_text'].fillna('').tolist()
            
            # Vectorize the text
            X = vectorizer.fit_transform(documents)
            
            # Get feature names
            feature_names = vectorizer.get_feature_names_out()
            
            # Create and fit LDA model
            lda = LatentDirichletAllocation(
                n_components=n_topics,
                max_iter=10,
                learning_method='online',
                random_state=42
            )
            
            lda.fit(X)
            
            # Extract top words for each topic
            topics = []
            for topic_idx, topic in enumerate(lda.components_):
                top_words_idx = topic.argsort()[:-n_words-1:-1]
                top_words = [feature_names[i] for i in top_words_idx]
                topics.append((topic_idx, top_words))
            
            # Store the results
            self.topic_results[place_id] = {
                'model': lda,
                'vectorizer': vectorizer,
                'topics': topics
            }
            
            # Get topic distribution for each document
            topic_distribution = lda.transform(X)
            
            # Calculate topic diversity (entropy of topic distribution)
            avg_topic_entropy = np.mean([entropy(doc_topics) for doc_topics in topic_distribution])
            
            # Update metrics with topic diversity
            idx = self.metrics_df[self.metrics_df['placeId'] == place_id].index
            if len(idx) > 0:
                self.metrics_df.at[idx[0], 'topic_diversity'] = avg_topic_entropy
        
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
            
            # Check for text similarity among reviews
            if len(place_reviews) >= 5:
                # Use TF-IDF to measure review similarity
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
                except:
                    similar_review_rate = 0
            else:
                similar_review_rate = 0
            
            # Create authenticity metrics
            place_authenticity = {
                'placeId': place_id,
                'is_polarized': is_polarized,
                'high_five_star_rate': high_five_star,
                'has_review_clusters': has_review_clusters,
                'cluster_count': cluster_count,
                'largest_cluster_size': largest_cluster,
                'similar_review_rate': similar_review_rate,
                'authenticity_concerns': is_polarized or high_five_star or has_review_clusters or similar_review_rate > 0.1
            }
            
            authenticity_metrics.append(place_authenticity)
        
        self.authenticity_df = pd.DataFrame(authenticity_metrics)
        
        # Merge with main metrics
        self.metrics_df = pd.merge(self.metrics_df, self.authenticity_df, on='placeId', how='left')
        print(f"Review authenticity analysis completed for {len(self.authenticity_df)} establishments")
        return self.authenticity_df
    
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
        weights = {
            # Rating and volume (30%)
            'avg_rating': 15,
            'total_reviews': 7.5,
            'rating_consistency': 7.5,
            
            # Sentiment (20%)
            'avg_sentiment': 10,
            'positive_review_pct': 5,
            'sentiment_trend': 5,
            
            # Service quality aspects (15%)
            'service_sentiment': 5,
            'quality_sentiment': 5,
            'response_rate': 5,
            
            # Customer loyalty (15%)
            'customer_loyalty_score': 7.5,
            'will_return_rate': 7.5,
            
            # Temporal factors (10%)
            'recent_reviews_pct': 5,
            'rating_trend': 5,
            
            # Authenticity and trustworthiness (10%)
            'verified_reviewer_pct': 5,
            'authenticity_concerns': -5  # Negative weight
        }
        
        # Create a copy of the metrics dataframe for scoring
        score_df = self.metrics_df.copy()
        
        # Normalize numerical features to 0-1 scale
        for column in weights.keys():
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
        
        for column, weight in weights.items():
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
        self.analyze_comparative_references()
        self.analyze_customer_journey()
        self.analyze_complaint_resolution()
        self.create_composite_score()
        self.generate_insights()
        
        return self.results_df
    
    def save_results(self, output_dir='.'):
        """Save analysis results to files."""
        print("Saving results...")
        
        # Create output directory if it doesn't exist
        import os
        os.makedirs(output_dir, exist_ok=True)
        
        # Save main results
        if self.results_df is not None:
            self.results_df.to_csv(f"{output_dir}/establishment_rankings.csv", index=False)
            print(f"Rankings saved to {output_dir}/establishment_rankings.csv")
        
        # Save full metrics
        if self.metrics_df is not None:
            self.metrics_df.to_csv(f"{output_dir}/establishment_metrics.csv", index=False)
            print(f"Full metrics saved to {output_dir}/establishment_metrics.csv")
        
        # Save insights as JSON
        if self.insights is not None:
            import json
            with open(f"{output_dir}/establishment_insights.json", 'w') as f:
                json.dump(self.insights, f, indent=2)
            print(f"Insights saved to {output_dir}/establishment_insights.json")
        
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