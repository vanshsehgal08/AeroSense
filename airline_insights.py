"""
Airline Insights Module
Scrapes and analyzes live reviews for airlines, providing actionable insights
"""

import requests
from bs4 import BeautifulSoup
import pandas as pd
import re
from collections import Counter
from datetime import datetime
import time


class AirlineReviewScraper:
    """Scrape reviews from various airline review platforms"""
    
    def __init__(self):
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.5',
        }
        
        # Airline-specific URLs for Trustpilot
        self.airline_urls = {
            'indigo': 'https://www.trustpilot.com/review/www.goindigo.in',
            'air_india': 'https://www.trustpilot.com/review/www.airindia.in',
            'spicejet': 'https://www.trustpilot.com/review/www.spicejet.com',
            'vistara': 'https://www.trustpilot.com/review/www.airvistara.com',
            'airasia': 'https://www.trustpilot.com/review/www.airasia.com',
            'qatar_airways': 'https://www.trustpilot.com/review/www.qatarairways.com',
            'monarch_air_group': 'https://www.trustpilot.com/review/monarchairgroup.com',
        }
    
    def scrape_trustpilot_reviews(self, airline_name, max_pages=5):
        """Scrape reviews from Trustpilot for a specific airline"""
        airline_key = airline_name.lower().replace(' ', '_')
        
        if airline_key not in self.airline_urls:
            # Try to construct URL
            base_url = f"https://www.trustpilot.com/review/www.{airline_name.lower().replace(' ', '')}.com"
        else:
            base_url = self.airline_urls[airline_key]
        
        reviews = []
        ratings = []
        dates = []
        
        print(f"Scraping reviews from: {base_url}")
        
        for page in range(1, max_pages + 1):
            try:
                if page == 1:
                    url = base_url
                else:
                    url = f"{base_url}?page={page}"
                
                response = requests.get(url, headers=self.headers, timeout=10)
                
                if response.status_code != 200:
                    print(f"Status code {response.status_code} on page {page}")
                    break
                
                soup = BeautifulSoup(response.content, 'html.parser')
                
                # Try different selectors for Trustpilot
                review_blocks = soup.find_all("div", class_="styles_reviewContent__tuXiN")
                if not review_blocks:
                    review_blocks = soup.find_all("section", {"data-review-id": True})
                
                if not review_blocks:
                    print(f"No reviews found on page {page}")
                    break
                
                print(f"Found {len(review_blocks)} reviews on page {page}")
                
                for block in review_blocks:
                    # Extract review text
                    review_text = ""
                    p_tag = block.find("p", class_="typography_body-l__v5JLj")
                    if not p_tag:
                        p_tag = block.find("p")
                    if p_tag:
                        review_text = p_tag.text.strip()
                    
                    if not review_text or review_text == "No review text":
                        continue
                    
                    # Extract rating
                    rating = -1
                    rating_div = block.find_previous("div", class_="styles_reviewHeader__DzoAZ")
                    if rating_div:
                        rating = int(rating_div.get("data-service-review-rating", -1))
                    else:
                        # Try alternative method
                        stars = block.find_all("div", class_="star-rating")
                        if stars:
                            rating = len(stars)
                    
                    # Extract date if available
                    date = datetime.now().strftime("%Y-%m-%d")
                    date_elem = block.find("time")
                    if date_elem:
                        date = date_elem.get("datetime", date)
                    
                    reviews.append(review_text)
                    ratings.append(rating if rating > 0 else 3)
                    dates.append(date)
                
                # Be respectful - add delay between requests
                time.sleep(1)
                
            except Exception as e:
                print(f"Error scraping page {page}: {e}")
                continue
        
        df = pd.DataFrame({
            'Review': reviews,
            'Rating': ratings,
            'Date': dates,
            'Airline': airline_name
        })
        
        return df
    
    def scrape_custom_url(self, url, max_pages=5):
        """Scrape reviews from a custom URL"""
        reviews = []
        ratings = []
        dates = []
        
        for page in range(1, max_pages + 1):
            try:
                if page == 1:
                    page_url = url
                else:
                    page_url = f"{url}?page={page}"
                
                response = requests.get(page_url, headers=self.headers, timeout=10)
                
                if response.status_code != 200:
                    break
                
                soup = BeautifulSoup(response.content, 'html.parser')
                review_blocks = soup.find_all("div", class_="styles_reviewContent__tuXiN")
                
                if not review_blocks:
                    break
                
                for block in review_blocks:
                    p_tag = block.find("p", class_="typography_body-l__v5JLj")
                    if p_tag:
                        review_text = p_tag.text.strip()
                        if review_text and review_text != "No review text":
                            reviews.append(review_text)
                            
                            # Try to get rating
                            rating_div = block.find_previous("div", class_="styles_reviewHeader__DzoAZ")
                            rating = int(rating_div.get("data-service-review-rating", 3)) if rating_div else 3
                            ratings.append(rating)
                            dates.append(datetime.now().strftime("%Y-%m-%d"))
                
                time.sleep(1)
                
            except Exception as e:
                print(f"Error: {e}")
                continue
        
        return pd.DataFrame({
            'Review': reviews,
            'Rating': ratings,
            'Date': dates
        })


class AirlineInsightsAnalyzer:
    """Analyze airline reviews and generate insights"""
    
    def __init__(self, nlp_processor=None):
        self.nlp_processor = nlp_processor
        
        # Common airline issues keywords
        self.issue_keywords = {
            'delays': ['delay', 'late', 'cancelled', 'rescheduled', 'missed', 'waiting'],
            'customer_service': ['service', 'staff', 'rude', 'unhelpful', 'unprofessional', 'unresponsive'],
            'baggage': ['luggage', 'baggage', 'lost', 'damaged', 'missing', 'bag'],
            'booking': ['booking', 'reservation', 'payment', 'refund', 'website', 'app'],
            'comfort': ['seat', 'comfort', 'legroom', 'space', 'cramped', 'uncomfortable'],
            'food': ['food', 'meal', 'snack', 'beverage', 'quality'],
            'pricing': ['price', 'expensive', 'cheap', 'cost', 'fare', 'overpriced'],
            'cleanliness': ['clean', 'dirty', 'hygiene', 'sanitized', 'messy'],
            'safety': ['safety', 'security', 'concern', 'unsafe'],
            'entertainment': ['entertainment', 'wifi', 'charger', 'screen', 'amenities']
        }
        
        self.positive_keywords = {
            'punctual': ['on time', 'punctual', 'timely', 'schedule'],
            'friendly_staff': ['friendly', 'helpful', 'polite', 'courteous', 'professional'],
            'clean': ['clean', 'hygienic', 'neat', 'tidy'],
            'comfortable': ['comfortable', 'spacious', 'good seat', 'comfort'],
            'value': ['value', 'worth', 'affordable', 'reasonable'],
            'food_quality': ['good food', 'tasty', 'delicious', 'quality meal']
        }
    
    def analyze_airline_reviews(self, df, airline_name):
        """Generate comprehensive insights for an airline"""
        if df.empty:
            return {
                'error': 'No reviews found',
                'airline': airline_name
            }
        
        # Assign sentiment based on rating
        df['Sentiment'] = df['Rating'].apply(lambda x: 
            'Positive' if x >= 4 else 'Negative' if x <= 2 else 'Neutral'
        )
        
        insights = {
            'airline': airline_name,
            'total_reviews': len(df),
            'summary': self._generate_summary(df),
            'sentiment_distribution': df['Sentiment'].value_counts().to_dict(),
            'rating_distribution': df['Rating'].value_counts().to_dict(),
            'top_issues': self._identify_issues(df),
            'positive_aspects': self._identify_positives(df),
            'difficulties': self._identify_difficulties(df),
            'recommendations': self._generate_recommendations(df),
            'trends': self._analyze_trends(df),
            'category_breakdown': self._category_breakdown(df)
        }
        
        return insights
    
    def _generate_summary(self, df):
        """Generate executive summary"""
        total = len(df)
        avg_rating = df['Rating'].mean()
        positive_pct = (df['Sentiment'] == 'Positive').sum() / total * 100
        negative_pct = (df['Sentiment'] == 'Negative').sum() / total * 100
        
        return {
            'average_rating': float(avg_rating),
            'positive_percentage': float(positive_pct),
            'negative_percentage': float(negative_pct),
            'overall_sentiment': 'Positive' if avg_rating >= 3.5 else 'Negative' if avg_rating < 2.5 else 'Mixed'
        }
    
    def _identify_issues(self, df):
        """Identify common issues and failures"""
        negative_reviews = df[df['Sentiment'] == 'Negative']['Review'].tolist()
        
        issue_counts = {}
        for category, keywords in self.issue_keywords.items():
            count = 0
            for review in negative_reviews:
                review_lower = review.lower()
                if any(keyword in review_lower for keyword in keywords):
                    count += 1
            if count > 0:
                issue_counts[category.replace('_', ' ').title()] = count
        
        # Sort by frequency
        sorted_issues = sorted(issue_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'top_5_issues': dict(sorted_issues[:5]),
            'all_issues': issue_counts,
            'total_negative_reviews': len(negative_reviews)
        }
    
    def _identify_positives(self, df):
        """Identify positive aspects"""
        positive_reviews = df[df['Sentiment'] == 'Positive']['Review'].tolist()
        
        positive_counts = {}
        for category, keywords in self.positive_keywords.items():
            count = 0
            for review in positive_reviews:
                review_lower = review.lower()
                if any(keyword in review_lower for keyword in keywords):
                    count += 1
            if count > 0:
                positive_counts[category.replace('_', ' ').title()] = count
        
        sorted_positives = sorted(positive_counts.items(), key=lambda x: x[1], reverse=True)
        
        return {
            'top_5_positives': dict(sorted_positives[:5]),
            'all_positives': positive_counts
        }
    
    def _identify_difficulties(self, df):
        """Identify difficulties and pain points"""
        difficulties = []
        
        # Check for common difficulty phrases
        difficulty_phrases = [
            'difficult', 'hard', 'struggle', 'problem', 'issue', 
            'challenge', 'trouble', 'complicated', 'frustrating'
        ]
        
        for idx, row in df.iterrows():
            review_lower = row['Review'].lower()
            if any(phrase in review_lower for phrase in difficulty_phrases):
                difficulties.append({
                    'review': row['Review'][:200] + '...' if len(row['Review']) > 200 else row['Review'],
                    'rating': int(row['Rating']),
                    'sentiment': row['Sentiment']
                })
        
        return {
            'total_difficulties': len(difficulties),
            'sample_difficulties': difficulties[:10]
        }
    
    def _generate_recommendations(self, df):
        """Generate actionable recommendations"""
        recommendations = []
        
        # Analyze issues and generate recommendations
        issues = self._identify_issues(df)
        top_issues = issues['top_5_issues']
        
        recommendation_map = {
            'Delays': 'Improve flight punctuality and communication about delays',
            'Customer Service': 'Train staff in customer service and responsiveness',
            'Baggage': 'Implement better baggage tracking and handling procedures',
            'Booking': 'Improve website/app functionality and booking process',
            'Comfort': 'Enhance seat comfort and cabin space',
            'Food': 'Improve meal quality and options',
            'Pricing': 'Review pricing strategy and transparency',
            'Cleanliness': 'Maintain higher cabin cleanliness standards',
            'Safety': 'Communicate safety measures clearly to passengers',
            'Entertainment': 'Upgrade in-flight entertainment options'
        }
        
        for issue, count in top_issues.items():
            if issue in recommendation_map:
                recommendations.append({
                    'priority': 'High' if count > issues['total_negative_reviews'] * 0.3 else 'Medium',
                    'issue': issue,
                    'affected_reviews': count,
                    'recommendation': recommendation_map[issue]
                })
        
        return recommendations
    
    def _analyze_trends(self, df):
        """Analyze trends over time if dates are available"""
        if 'Date' not in df.columns:
            return {'message': 'Date information not available'}
        
        try:
            df['Date'] = pd.to_datetime(df['Date'])
            df = df.sort_values('Date')
            
            # Monthly trend
            monthly = df.groupby(df['Date'].dt.to_period('M')).agg({
                'Rating': 'mean',
                'Sentiment': lambda x: (x == 'Positive').sum() / len(x) * 100
            }).reset_index()
            
            return {
                'monthly_trend': [
                    {
                        'month': str(period),
                        'avg_rating': float(row['Rating']),
                        'positive_percentage': float(row['Sentiment'])
                    }
                    for period, row in zip(monthly['Date'], monthly.itertuples())
                ]
            }
        except:
            return {'message': 'Could not analyze trends'}
    
    def _category_breakdown(self, df):
        """Breakdown by complaint categories"""
        categories = {}
        
        for category, keywords in self.issue_keywords.items():
            count = 0
            for review in df['Review']:
                review_lower = review.lower()
                if any(keyword in review_lower for keyword in keywords):
                    count += 1
            if count > 0:
                categories[category.replace('_', ' ').title()] = count
        
        return categories

