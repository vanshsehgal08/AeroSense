import React, { useState, useEffect } from 'react';
import { sentimentAPI } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, Cell } from 'recharts';
import './SentimentPredictor.css';

const SentimentPredictor = () => {
  const [review, setReview] = useState('');
  const [result, setResult] = useState(null);
  const [explanation, setExplanation] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [loading, setLoading] = useState(false);
  const [explainLoading, setExplainLoading] = useState(false);
  const [error, setError] = useState(null);
  const [showExplanation, setShowExplanation] = useState(true);

  useEffect(() => {
    loadFeatureImportance();
  }, []);

  const loadFeatureImportance = async () => {
    try {
      const response = await sentimentAPI.getFeatureImportance();
      if (response.data && response.data.feature_importance && response.data.feature_importance.length > 0) {
        setFeatureImportance(response.data);
      } else {
        console.warn('Feature importance data is empty or unavailable');
        setFeatureImportance(null);
      }
    } catch (err) {
      console.error('Failed to load feature importance:', err);
      setFeatureImportance(null);
    }
  };

  const handlePredict = async () => {
    if (!review.trim()) {
      setError('Please enter a review');
      return;
    }

    setLoading(true);
    setError(null);
    setResult(null);
    setExplanation(null);

    try {
      // Get prediction
      const response = await sentimentAPI.predictSentiment(review);
      setResult(response.data);

      // Automatically get explanation if enabled
      if (showExplanation) {
        setExplainLoading(true);
        try {
          const explainResponse = await sentimentAPI.explainPrediction(review, 10);
          setExplanation(explainResponse.data);
        } catch (err) {
          console.error('Failed to get explanation:', err);
        } finally {
          setExplainLoading(false);
        }
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to predict sentiment');
    } finally {
      setLoading(false);
    }
  };

  const getSentimentColor = (sentiment) => {
    switch (sentiment) {
      case 'Positive':
        return '#10b981';
      case 'Negative':
        return '#ef4444';
      case 'Neutral':
        return '#f59e0b';
      default:
        return '#6b7280';
    }
  };

  const getSentimentIcon = (sentiment) => {
    switch (sentiment) {
      case 'Positive':
        return '😊';
      case 'Negative':
        return '😞';
      case 'Neutral':
        return '😐';
      default:
        return '🤔';
    }
  };

  return (
    <div className="sentiment-predictor">
      <h2>Predict Sentiment</h2>
      <p className="subtitle">Enter a review text to analyze its sentiment</p>

      <div className="predictor-container">
        <div className="input-section">
          <textarea
            className="review-input"
            placeholder="Enter your review here..."
            value={review}
            onChange={(e) => setReview(e.target.value)}
            rows={8}
          />
          <button
            className="predict-button"
            onClick={handlePredict}
            disabled={loading || !review.trim()}
          >
            {loading ? 'Analyzing...' : 'Predict Sentiment'}
          </button>
        </div>

        {error && (
          <div className="error-message">
            {error}
          </div>
        )}

        {result && (
          <div className="result-section">
            <div
              className="sentiment-card"
              style={{ borderColor: getSentimentColor(result.sentiment) }}
            >
              <div className="sentiment-header">
                <span className="sentiment-icon">{getSentimentIcon(result.sentiment)}</span>
                <h3 style={{ color: getSentimentColor(result.sentiment) }}>
                  {result.sentiment}
                </h3>
              </div>

              <div className="probability-section">
                <h4>Confidence Scores:</h4>
                <div className="probability-bars">
                  {Object.entries(result.probabilities).map(([sentiment, prob]) => (
                    <div key={sentiment} className="probability-item">
                      <div className="probability-label">
                        <span>{sentiment}</span>
                        <span className="probability-value">
                          {(prob * 100).toFixed(2)}%
                        </span>
                      </div>
                      <div className="probability-bar-container">
                        <div
                          className="probability-bar"
                          style={{
                            width: `${prob * 100}%`,
                            backgroundColor: getSentimentColor(sentiment),
                          }}
                        />
                      </div>
                    </div>
                  ))}
                </div>
              </div>

              {result.category && (
                <div className="category-section">
                  <h4>Category:</h4>
                  <span className="category-badge">{result.category}</span>
                </div>
              )}
            </div>

            {/* Explanation Section */}
            {showExplanation && (
              <div className="explanation-section">
                <div className="explanation-header">
                  <h3>🔍 Prediction Explanation</h3>
                  <label className="toggle-label">
                    <input
                      type="checkbox"
                      checked={showExplanation}
                      onChange={(e) => setShowExplanation(e.target.checked)}
                    />
                    Auto-explain
                  </label>
                </div>

                {explainLoading && (
                  <div className="loading-explanation">Generating explanation...</div>
                )}

                {explanation && !explainLoading && (
                  <div className="explanation-content">
                    {explanation.top_features && explanation.top_features.length > 0 && (
                      <div className="feature-contributions">
                        <h4>Top Contributing Features</h4>
                        <div className="features-list">
                          {explanation.top_features.map((feature, idx) => (
                            <div key={idx} className="feature-item">
                              <div className="feature-name">{feature.feature}</div>
                              <div className={`feature-contribution ${feature.contribution > 0 ? 'positive' : 'negative'}`}>
                                {feature.contribution > 0 ? '+' : ''}{feature.contribution.toFixed(4)}
                              </div>
                            </div>
                          ))}
                        </div>
                      </div>
                    )}

                    {explanation.positive_contributors && explanation.positive_contributors.length > 0 && (
                      <div className="contributors-section">
                        <h4>Positive Contributors</h4>
                        <div className="contributors-list">
                          {explanation.positive_contributors.map((f, idx) => (
                            <span key={idx} className="contributor-badge positive">
                              {f.feature}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}

                    {explanation.negative_contributors && explanation.negative_contributors.length > 0 && (
                      <div className="contributors-section">
                        <h4>Negative Contributors</h4>
                        <div className="contributors-list">
                          {explanation.negative_contributors.map((f, idx) => (
                            <span key={idx} className="contributor-badge negative">
                              {f.feature}
                            </span>
                          ))}
                        </div>
                      </div>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Feature Importance Chart */}
            {featureImportance && featureImportance.feature_importance && featureImportance.feature_importance.length > 0 ? (
              <div className="feature-importance-section">
                <h3>📊 Global Feature Importance</h3>
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart
                      data={featureImportance.feature_importance.slice(0, 15).map((f) => ({
                        name: f.feature.length > 20 ? f.feature.substring(0, 20) + '...' : f.feature,
                        importance: Math.abs(f.importance),
                        positive: f.importance > 0,
                        originalImportance: f.importance,
                      }))}
                      layout="vertical"
                    >
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis type="number" />
                      <YAxis dataKey="name" type="category" width={150} />
                      <Tooltip />
                      <Bar dataKey="importance">
                        {featureImportance.feature_importance.slice(0, 15).map((f, index) => (
                          <Cell key={`cell-${index}`} fill={f.importance > 0 ? '#10b981' : '#ef4444'} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
              </div>
            ) : (
              <div className="feature-importance-section">
                <h3>📊 Global Feature Importance</h3>
                <div className="empty-state">
                  <p>Feature importance data is not available. Please train the model first.</p>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default SentimentPredictor;

