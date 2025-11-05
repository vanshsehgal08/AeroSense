import React, { useState } from 'react';
import { sentimentAPI } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line } from 'recharts';
import './AirlineInsights.css';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d'];

const AirlineInsights = () => {
  const [loading, setLoading] = useState(false);
  const [insights, setInsights] = useState(null);
  const [error, setError] = useState(null);
  const [options, setOptions] = useState({
    airline_name: 'indigo',
    max_pages: 5,
    use_custom_url: false,
    custom_url: '',
  });

  const handleAnalyze = async () => {
    setLoading(true);
    setError(null);
    setInsights(null);

    try {
      const response = await sentimentAPI.scrapeAndAnalyze({
        airline_name: options.airline_name,
        url: options.use_custom_url ? options.custom_url : '',
        max_pages: options.max_pages,
      });
      setInsights(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to analyze airline reviews');
    } finally {
      setLoading(false);
    }
  };

  const prepareIssuesData = () => {
    if (!insights?.top_issues?.top_5_issues) return [];
    return Object.entries(insights.top_issues.top_5_issues).map(([issue, count]) => ({
      name: issue,
      value: count,
    }));
  };

  const preparePositivesData = () => {
    if (!insights?.positive_aspects?.top_5_positives) return [];
    return Object.entries(insights.positive_aspects.top_5_positives).map(([aspect, count]) => ({
      name: aspect,
      value: count,
    }));
  };

  const prepareSentimentData = () => {
    if (!insights?.sentiment_distribution) return [];
    return Object.entries(insights.sentiment_distribution).map(([sentiment, count]) => ({
      name: sentiment,
      value: count,
    }));
  };

  const prepareRatingData = () => {
    if (!insights?.rating_distribution) return [];
    return Object.entries(insights.rating_distribution)
      .sort(([a], [b]) => Number(b) - Number(a))
      .map(([rating, count]) => ({
        name: `Rating ${rating}`,
        value: count,
      }));
  };

  const prepareTrendData = () => {
    if (!insights?.trends?.monthly_trend) return [];
    return insights.trends.monthly_trend.map(item => ({
      month: item.month,
      rating: item.avg_rating,
      positive: item.positive_percentage,
    }));
  };

  const [activeView, setActiveView] = useState('overview');

  return (
    <div className="airline-insights">
      <div className="insights-header">
        <div>
          <h2>✈️ Airline Insights</h2>
          <p className="subtitle">Analyze live reviews and generate actionable insights</p>
        </div>
        <div className="quick-actions">
          <div className="quick-options">
              <select
                className="airline-select"
                value={options.airline_name}
                onChange={(e) => setOptions({ ...options, airline_name: e.target.value })}
                disabled={options.use_custom_url || loading}
              >
                <option value="indigo">Indigo</option>
                <option value="air_india">Air India</option>
                <option value="spicejet">SpiceJet</option>
                <option value="vistara">Vistara</option>
                <option value="airasia">AirAsia</option>
                <option value="qatar_airways">Qatar Airways</option>
                <option value="monarch_air_group">Monarch Air Group</option>
              </select>
            <input
              type="range"
              className="pages-slider"
              min="1"
              max="10"
              value={options.max_pages}
              onChange={(e) => setOptions({ ...options, max_pages: parseInt(e.target.value) })}
              disabled={loading}
              title={`${options.max_pages} pages`}
            />
            <span className="pages-label">{options.max_pages} pages</span>
          </div>
          <button
            className="analyze-button"
            onClick={handleAnalyze}
            disabled={loading}
          >
            {loading ? 'Analyzing...' : '🚀 Analyze'}
          </button>
        </div>
      </div>

      <div className="insights-container">
        <div className="custom-url-toggle">
          <label className="checkbox-label">
            <input
              type="checkbox"
              checked={options.use_custom_url}
              onChange={(e) => setOptions({ ...options, use_custom_url: e.target.checked })}
              disabled={loading}
            />
            Use Custom URL
          </label>
          {options.use_custom_url && (
            <input
              type="text"
              className="custom-url-input"
              value={options.custom_url}
              onChange={(e) => setOptions({ ...options, custom_url: e.target.value })}
              placeholder="https://www.trustpilot.com/review/..."
              disabled={loading}
            />
          )}
        </div>

        {error && (
          <div className="error-message">{error}</div>
        )}

        {insights && (
          <div className="insights-results">
            {/* Summary Card - Compact */}
            <div className="summary-card">
              <div className="summary-header">
                <h3>{insights.airline}</h3>
                <span className="sentiment-badge">{insights.summary?.overall_sentiment || 'N/A'}</span>
              </div>
              <div className="summary-stats">
                <div className="stat-item">
                  <span className="stat-value">{insights.total_reviews}</span>
                  <span className="stat-label">Reviews</span>
                </div>
                <div className="stat-item">
                  <span className="stat-value">{insights.summary?.average_rating?.toFixed(1) || 'N/A'}</span>
                  <span className="stat-label">Avg Rating</span>
                </div>
                <div className="stat-item positive">
                  <span className="stat-value">{insights.summary?.positive_percentage?.toFixed(1) || 'N/A'}%</span>
                  <span className="stat-label">Positive</span>
                </div>
                <div className="stat-item negative">
                  <span className="stat-value">{insights.summary?.negative_percentage?.toFixed(1) || 'N/A'}%</span>
                  <span className="stat-label">Negative</span>
                </div>
              </div>
            </div>

            {/* View Tabs */}
            <div className="view-tabs">
              <button
                className={activeView === 'overview' ? 'active' : ''}
                onClick={() => setActiveView('overview')}
              >
                Overview
              </button>
              <button
                className={activeView === 'issues' ? 'active' : ''}
                onClick={() => setActiveView('issues')}
              >
                Issues
              </button>
              <button
                className={activeView === 'recommendations' ? 'active' : ''}
                onClick={() => setActiveView('recommendations')}
              >
                Recommendations
              </button>
            </div>

            {/* Overview Tab */}
            {activeView === 'overview' && (
              <>
                <div className="overview-grid">
                  {/* Sentiment & Rating */}
                  <div className="insight-card">
                    <h4>Sentiment Distribution</h4>
                    <ResponsiveContainer width="100%" height={200}>
                      <PieChart>
                        <Pie
                          data={prepareSentimentData()}
                          cx="50%"
                          cy="50%"
                          labelLine={false}
                          label={({ percent }) => `${(percent * 100).toFixed(0)}%`}
                          outerRadius={70}
                          fill="#8884d8"
                          dataKey="value"
                        >
                          {prepareSentimentData().map((entry, index) => (
                            <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                          ))}
                        </Pie>
                        <Tooltip />
                      </PieChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="insight-card">
                    <h4>Rating Distribution</h4>
                    <ResponsiveContainer width="100%" height={200}>
                      <BarChart data={prepareRatingData()}>
                        <XAxis dataKey="name" />
                        <YAxis />
                        <Tooltip />
                        <Bar dataKey="value" fill="#667eea" />
                      </BarChart>
                    </ResponsiveContainer>
                  </div>
                </div>

                {/* Positive Aspects */}
                {insights.positive_aspects && insights.positive_aspects.top_5_positives && (
                  <div className="insight-card">
                    <h4>✅ Positive Aspects</h4>
                    <div className="aspects-grid">
                      {Object.entries(insights.positive_aspects.top_5_positives).map(([aspect, count]) => (
                        <div key={aspect} className="aspect-item positive">
                          <span className="aspect-name">{aspect}</span>
                          <span className="aspect-count">{count}</span>
                        </div>
                      ))}
                    </div>
                  </div>
                )}
              </>
            )}

            {/* Issues Tab */}
            {activeView === 'issues' && insights.top_issues && (
              <div className="insight-card">
                <h4>🚨 Top Issues</h4>
                <div className="chart-container">
                  <ResponsiveContainer width="100%" height={300}>
                    <BarChart data={prepareIssuesData()}>
                      <CartesianGrid strokeDasharray="3 3" />
                      <XAxis dataKey="name" />
                      <YAxis />
                      <Tooltip />
                      <Bar dataKey="value" fill="#ef4444">
                        {prepareIssuesData().map((entry, index) => (
                          <Cell key={`cell-${index}`} fill={COLORS[index % COLORS.length]} />
                        ))}
                      </Bar>
                    </BarChart>
                  </ResponsiveContainer>
                </div>
                <div className="issues-summary">
                  <span>Total Negative Reviews: <strong>{insights.top_issues.total_negative_reviews}</strong></span>
                </div>
              </div>
            )}

            {/* Recommendations Tab */}
            {activeView === 'recommendations' && insights.recommendations && insights.recommendations.length > 0 && (
              <div className="insight-card">
                <h4>💡 Actionable Recommendations</h4>
                <div className="recommendations-list">
                  {insights.recommendations.map((rec, idx) => (
                    <div key={idx} className={`recommendation-card ${rec.priority.toLowerCase()}`}>
                      <div className="rec-header">
                        <span className="priority-badge">{rec.priority}</span>
                        <span className="issue-name">{rec.issue}</span>
                        <span className="affected-count">{rec.affected_reviews} reviews</span>
                      </div>
                      <div className="rec-text">{rec.recommendation}</div>
                    </div>
                  ))}
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default AirlineInsights;

