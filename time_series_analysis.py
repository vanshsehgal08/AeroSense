"""
Time Series Analysis Module
Analyzes sentiment trends over time
"""

import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import json

class TimeSeriesAnalyzer:
    """Analyzes sentiment trends and patterns over time"""
    
    def __init__(self):
        pass
    
    def analyze_sentiment_trends(self, df, date_column='Date', sentiment_column='Sentiment'):
        """Analyze sentiment trends over time"""
        try:
            # Ensure date column is datetime
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            else:
                # Create date column from index if available
                df[date_column] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
            
            # Group by date and sentiment
            df['date_only'] = df[date_column].dt.date
            
            trend_data = df.groupby(['date_only', sentiment_column]).size().reset_index(name='count')
            trend_pivot = trend_data.pivot(index='date_only', columns=sentiment_column, values='count').fillna(0)
            
            # Calculate daily sentiment percentages
            daily_totals = trend_pivot.sum(axis=1)
            trend_percentages = trend_pivot.div(daily_totals, axis=0) * 100
            
            # Calculate moving averages
            window = min(7, len(trend_pivot) // 3)  # 7-day moving average
            if window > 1:
                moving_avg = trend_pivot.rolling(window=window).mean()
            else:
                moving_avg = trend_pivot
            
            return {
                'daily_counts': trend_pivot.to_dict('index'),
                'daily_percentages': trend_percentages.to_dict('index'),
                'moving_averages': moving_avg.to_dict('index'),
                'date_range': {
                    'start': str(trend_pivot.index.min()),
                    'end': str(trend_pivot.index.max())
                }
            }
        except Exception as e:
            return {'error': str(e)}
    
    def detect_sentiment_shifts(self, df, date_column='Date', sentiment_column='Sentiment', window=7):
        """Detect significant sentiment shifts"""
        try:
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            else:
                df[date_column] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
            
            df['date_only'] = df[date_column].dt.date
            
            # Calculate daily sentiment ratios
            daily_sentiment = df.groupby('date_only')[sentiment_column].apply(
                lambda x: (x == 'Positive').sum() / len(x) if len(x) > 0 else 0.5
            )
            
            # Calculate moving average and standard deviation
            ma = daily_sentiment.rolling(window=window).mean()
            std = daily_sentiment.rolling(window=window).std()
            
            # Detect shifts (values outside 2 standard deviations)
            shifts = []
            for date, value in daily_sentiment.items():
                if date in ma.index and date in std.index:
                    if pd.notna(ma[date]) and pd.notna(std[date]) and std[date] > 0:
                        z_score = abs((value - ma[date]) / std[date])
                        if z_score > 2:
                            shifts.append({
                                'date': str(date),
                                'sentiment_ratio': float(value),
                                'moving_average': float(ma[date]),
                                'z_score': float(z_score),
                                'type': 'positive_shift' if value > ma[date] else 'negative_shift'
                            })
            
            return shifts
        except Exception as e:
            return {'error': str(e)}
    
    def analyze_seasonal_patterns(self, df, date_column='Date', sentiment_column='Sentiment'):
        """Analyze seasonal patterns in sentiment"""
        try:
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            else:
                df[date_column] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
            
            # Extract temporal features
            df['month'] = df[date_column].dt.month
            df['day_of_week'] = df[date_column].dt.dayofweek
            df['hour'] = df[date_column].dt.hour if 'hour' in df.columns else 12  # Default if no time
            
            patterns = {}
            
            # Monthly patterns
            monthly = df.groupby('month')[sentiment_column].apply(
                lambda x: (x == 'Positive').sum() / len(x) if len(x) > 0 else 0
            )
            patterns['monthly'] = monthly.to_dict()
            
            # Day of week patterns
            weekly = df.groupby('day_of_week')[sentiment_column].apply(
                lambda x: (x == 'Positive').sum() / len(x) if len(x) > 0 else 0
            )
            patterns['weekly'] = weekly.to_dict()
            
            return patterns
        except Exception as e:
            return {'error': str(e)}
    
    def forecast_sentiment(self, df, date_column='Date', sentiment_column='Sentiment', days_ahead=7):
        """Simple sentiment forecasting using trend"""
        try:
            if date_column in df.columns:
                df[date_column] = pd.to_datetime(df[date_column], errors='coerce')
            else:
                df[date_column] = pd.date_range(start='2024-01-01', periods=len(df), freq='D')
            
            df['date_only'] = df[date_column].dt.date
            
            # Calculate daily positive ratio
            daily_ratio = df.groupby('date_only')[sentiment_column].apply(
                lambda x: (x == 'Positive').sum() / len(x) if len(x) > 0 else 0.5
            )
            
            # Simple linear trend
            if len(daily_ratio) > 1:
                x = np.arange(len(daily_ratio))
                y = daily_ratio.values
                
                # Fit linear trend
                coeffs = np.polyfit(x, y, 1)
                trend_line = np.poly1d(coeffs)
                
                # Forecast
                future_dates = pd.date_range(start=daily_ratio.index[-1], periods=days_ahead+1, freq='D')[1:]
                future_x = np.arange(len(daily_ratio), len(daily_ratio) + days_ahead)
                future_values = trend_line(future_x)
                
                forecast = [
                    {
                        'date': str(date.date()),
                        'predicted_positive_ratio': float(max(0, min(1, val)))
                    }
                    for date, val in zip(future_dates, future_values)
                ]
                
                return forecast
            else:
                return []
        except Exception as e:
            return {'error': str(e)}

