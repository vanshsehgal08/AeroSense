import React, { useState, useEffect } from 'react';
import { sentimentAPI } from '../services/api';
import './Statistics.css';

const Statistics = () => {
  const [stats, setStats] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    loadStats();
  }, []);

  const loadStats = async () => {
    setLoading(true);
    setError(null);
    try {
      const response = await sentimentAPI.getStats();
      setStats(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to load statistics');
    } finally {
      setLoading(false);
    }
  };

  if (loading) {
    return (
      <div className="statistics">
        <h2>Statistics</h2>
        <div className="loading">Loading statistics...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="statistics">
        <h2>Statistics</h2>
        <div className="error-message">{error}</div>
      </div>
    );
  }

  return (
    <div className="statistics">
      <div className="stats-header">
        <h2>Dataset Statistics</h2>
        <button className="refresh-button" onClick={loadStats}>
          🔄 Refresh
        </button>
      </div>

      {stats && (
        <div className="stats-container">
          {/* Model Info */}
          {stats.model && (
            <div className="stats-section">
              <h3>Model Information</h3>
              <div className="stats-grid">
                <div className="stat-card">
                  <div className="stat-value">
                    {stats.model.trained ? '✓' : '✗'}
                  </div>
                  <div className="stat-label">Model Status</div>
                </div>
                {stats.model.accuracy && (
                  <div className="stat-card">
                    <div className="stat-value">
                      {(stats.model.accuracy * 100).toFixed(2)}%
                    </div>
                    <div className="stat-label">Accuracy</div>
                  </div>
                )}
                {stats.model.training_samples && (
                  <div className="stat-card">
                    <div className="stat-value">{stats.model.training_samples}</div>
                    <div className="stat-label">Training Samples</div>
                  </div>
                )}
                {stats.model.test_samples && (
                  <div className="stat-card">
                    <div className="stat-value">{stats.model.test_samples}</div>
                    <div className="stat-label">Test Samples</div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Dataset Statistics */}
          {Object.entries(stats).filter(([key]) => key !== 'model').map(([key, data]) => {
            const datasetNames = {
              'sentiment': 'Sentiment Dataset (Original)',
              'cleaned': 'Cleaned Dataset (Preprocessed)',
              'categories': 'Categories Dataset (With Categories)'
            };
            
            const datasetDescriptions = {
              'sentiment': 'Original dataset with sentiment labels',
              'cleaned': 'Preprocessed text with cleaned reviews',
              'categories': 'Dataset with complaint categories assigned'
            };

            return (
              <div key={key} className="stats-section">
                <div className="dataset-header">
                  <h3>{datasetNames[key] || key.charAt(0).toUpperCase() + key.slice(1)} Dataset</h3>
                  <div className="dataset-description">
                    {datasetDescriptions[key] || 'Dataset statistics'}
                  </div>
                  {data.file_name && (
                    <div className="dataset-file">
                      📄 {data.file_name}
                    </div>
                  )}
                </div>

                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-value">{data.total_reviews}</div>
                    <div className="stat-label">Total Reviews</div>
                  </div>

                  {data.columns && (
                    <div className="stat-card info-card">
                      <div className="stat-value">{data.columns.length}</div>
                      <div className="stat-label">Columns</div>
                      <div className="stat-detail">{data.columns.join(', ')}</div>
                    </div>
                  )}

                  {/* Dataset-specific unique stats */}
                  {key === 'cleaned' && data.preprocessing_stats && (
                    <div className="stat-card info-card">
                      <div className="stat-value">
                        {data.preprocessing_stats.reduction_percentage.toFixed(1)}%
                      </div>
                      <div className="stat-label">Text Reduction</div>
                      <div className="stat-detail">
                        Avg: {data.preprocessing_stats.avg_original_length.toFixed(0)} → {data.preprocessing_stats.avg_cleaned_length.toFixed(0)} chars
                      </div>
                    </div>
                  )}

                  {key === 'categories' && data.total_categories && (
                    <div className="stat-card info-card">
                      <div className="stat-value">{data.total_categories}</div>
                      <div className="stat-label">Unique Categories</div>
                    </div>
                  )}

                  {data.sentiment_distribution && (
                    <>
                      {Object.entries(data.sentiment_distribution).map(([sentiment, count]) => (
                        <div key={sentiment} className="stat-card">
                          <div className="stat-value">{count}</div>
                          <div className="stat-label">{sentiment}</div>
                          {data.sentiment_percentages && data.sentiment_percentages[sentiment] && (
                            <div className="stat-detail">{data.sentiment_percentages[sentiment]}</div>
                          )}
                        </div>
                      ))}
                    </>
                  )}

                  {data.rating_distribution && (
                    <>
                      {Object.entries(data.rating_distribution)
                        .sort(([a], [b]) => Number(b) - Number(a))
                        .map(([rating, count]) => (
                          <div key={rating} className="stat-card">
                            <div className="stat-value">{count}</div>
                            <div className="stat-label">Rating {rating}</div>
                          </div>
                        ))}
                    </>
                  )}

                  {data.category_distribution && (
                    <>
                      <div className="stat-card full-width">
                        <div className="stat-label">Category Distribution</div>
                      </div>
                      {Object.entries(data.category_distribution)
                        .sort(([, a], [, b]) => b - a)
                        .slice(0, 8)
                        .map(([category, count]) => (
                          <div key={category} className="stat-card">
                            <div className="stat-value">{count}</div>
                            <div className="stat-label">{category}</div>
                          </div>
                        ))}
                    </>
                  )}
                </div>
              </div>
            );
          })}
        </div>
      )}
    </div>
  );
};

export default Statistics;

