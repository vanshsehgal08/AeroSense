import React, { useState } from 'react';
import { sentimentAPI } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer, PieChart, Pie, Cell, LineChart, Line, Legend } from 'recharts';
import './CompareAirlines.css';

const COLORS = ['#0088FE', '#00C49F', '#FFBB28', '#FF8042', '#8884d8', '#82ca9d', '#ffc658', '#ff7300'];

const CompareAirlines = () => {
  const [loading, setLoading] = useState(false);
  const [comparison, setComparison] = useState(null);
  const [error, setError] = useState(null);
  const [selectedAirlines, setSelectedAirlines] = useState(['indigo', 'air_india']);
  const [maxPages, setMaxPages] = useState(3);

  const availableAirlines = [
    { value: 'indigo', label: 'Indigo' },
    { value: 'air_india', label: 'Air India' },
    { value: 'spicejet', label: 'SpiceJet' },
    { value: 'vistara', label: 'Vistara' },
    { value: 'airasia', label: 'AirAsia' },
    { value: 'qatar_airways', label: 'Qatar Airways' },
    { value: 'monarch_air_group', label: 'Monarch Air Group' },
    { value: 'flyusa', label: 'FlyUSA' },
  ];

  const handleCompare = async () => {
    if (selectedAirlines.length < 2) {
      setError('Please select at least 2 airlines to compare');
      return;
    }

    setLoading(true);
    setError(null);
    setComparison(null);

    try {
      const response = await sentimentAPI.compareAirlines({
        airline_names: selectedAirlines,
        max_pages: maxPages,
      });
      setComparison(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to compare airlines');
    } finally {
      setLoading(false);
    }
  };

  const toggleAirline = (airlineValue) => {
    setSelectedAirlines(prev => {
      if (prev.includes(airlineValue)) {
        return prev.filter(a => a !== airlineValue);
      } else {
        return [...prev, airlineValue];
      }
    });
  };

  const prepareSentimentComparison = () => {
    if (!comparison?.comparison) return [];
    
    const airlines = Object.keys(comparison.comparison).filter(
      key => !comparison.comparison[key].error
    );
    
    const data = [];
    airlines.forEach(airline => {
      const dataPoint = { airline: availableAirlines.find(a => a.value === airline)?.label || airline };
      const sentiment = comparison.comparison[airline].sentiment_distribution || {};
      dataPoint.positive = sentiment.Positive || 0;
      dataPoint.negative = sentiment.Negative || 0;
      dataPoint.neutral = sentiment.Neutral || 0;
      data.push(dataPoint);
    });
    
    return data;
  };

  const preparePercentageComparison = () => {
    if (!comparison?.comparison) return [];
    
    const airlines = Object.keys(comparison.comparison).filter(
      key => !comparison.comparison[key].error
    );
    
    return airlines.map(airline => {
      const data = comparison.comparison[airline];
      const label = availableAirlines.find(a => a.value === airline)?.label || airline;
      return {
        airline: label,
        positive: data.positive_percentage || 0,
        negative: data.negative_percentage || 0,
        neutral: data.neutral_percentage || 0,
        rating: data.average_rating || 0,
        confidence: (data.model_confidence * 100) || 0,
      };
    });
  };

  const prepareRatingComparison = () => {
    if (!comparison?.comparison) return [];
    
    const airlines = Object.keys(comparison.comparison).filter(
      key => !comparison.comparison[key].error
    );
    
    return airlines.map(airline => {
      const data = comparison.comparison[airline];
      const label = availableAirlines.find(a => a.value === airline)?.label || airline;
      return {
        airline: label,
        rating: data.average_rating || 0,
        confidence: (data.model_confidence * 100) || 0,
      };
    });
  };

  const prepareIssuesComparison = () => {
    if (!comparison?.comparison) return [];
    
    const airlines = Object.keys(comparison.comparison).filter(
      key => !comparison.comparison[key].error
    );
    
    // Collect all unique issues
    const allIssues = new Set();
    airlines.forEach(airline => {
      const issues = comparison.comparison[airline].top_issues || {};
      Object.keys(issues).forEach(issue => allIssues.add(issue));
    });
    
    // Create data structure
    const data = [];
    allIssues.forEach(issue => {
      const dataPoint = { issue };
      airlines.forEach(airline => {
        const label = availableAirlines.find(a => a.value === airline)?.label || airline;
        const issues = comparison.comparison[airline].top_issues || {};
        dataPoint[label] = issues[issue] || 0;
      });
      data.push(dataPoint);
    });
    
    return data;
  };

  return (
    <div className="compare-airlines">
      <div className="compare-header">
        <div>
          <h2>🔀 Compare Airlines</h2>
          <p className="subtitle">Use AI model to analyze and compare multiple airlines side-by-side</p>
        </div>
        <div className="compare-controls">
          <div className="pages-control">
            <label>Pages per airline:</label>
            <input
              type="range"
              min="1"
              max="5"
              value={maxPages}
              onChange={(e) => setMaxPages(parseInt(e.target.value))}
              disabled={loading}
            />
            <span>{maxPages}</span>
          </div>
          <button
            className="compare-button"
            onClick={handleCompare}
            disabled={loading || selectedAirlines.length < 2}
          >
            {loading ? 'Comparing...' : '🚀 Compare Airlines'}
          </button>
        </div>
      </div>

      <div className="airline-selection">
        <h3>Select Airlines to Compare</h3>
        <div className="airline-checkboxes">
          {availableAirlines.map(airline => (
            <label key={airline.value} className="airline-checkbox">
              <input
                type="checkbox"
                checked={selectedAirlines.includes(airline.value)}
                onChange={() => toggleAirline(airline.value)}
                disabled={loading}
              />
              <span>{airline.label}</span>
            </label>
          ))}
        </div>
        <p className="selected-count">
          {selectedAirlines.length} airline{selectedAirlines.length !== 1 ? 's' : ''} selected
        </p>
      </div>

      {error && (
        <div className="error-message">{error}</div>
      )}

      {loading && (
        <div className="loading-message">
          <div className="spinner"></div>
          <p>Scraping reviews and analyzing with AI model... This may take a few moments.</p>
        </div>
      )}

      {comparison && (
        <div className="comparison-results">
          {/* Summary Cards */}
          {comparison.summary && (
            <div className="summary-section">
              <h3>📊 Comparison Summary</h3>
              <div className="summary-cards">
                {comparison.summary.best_positive && (
                  <div className="summary-card positive">
                    <div className="card-icon">🏆</div>
                    <div className="card-content">
                      <h4>Best Positive Sentiment</h4>
                      <p className="card-value">{comparison.summary.best_positive.airline}</p>
                      <p className="card-subvalue">{comparison.summary.best_positive.positive_percentage.toFixed(1)}%</p>
                    </div>
                  </div>
                )}
                {comparison.summary.best_rating && (
                  <div className="summary-card rating">
                    <div className="card-icon">⭐</div>
                    <div className="card-content">
                      <h4>Highest Rating</h4>
                      <p className="card-value">{comparison.summary.best_rating.airline}</p>
                      <p className="card-subvalue">{comparison.summary.best_rating.average_rating.toFixed(2)}</p>
                    </div>
                  </div>
                )}
                {comparison.summary.most_issues && (
                  <div className="summary-card issues">
                    <div className="card-icon">⚠️</div>
                    <div className="card-content">
                      <h4>Most Issues</h4>
                      <p className="card-value">{comparison.summary.most_issues.airline}</p>
                      <p className="card-subvalue">{comparison.summary.most_issues.negative_percentage.toFixed(1)}%</p>
                    </div>
                  </div>
                )}
                {comparison.summary.highest_confidence && (
                  <div className="summary-card confidence">
                    <div className="card-icon">🎯</div>
                    <div className="card-content">
                      <h4>Highest Model Confidence</h4>
                      <p className="card-value">{comparison.summary.highest_confidence.airline}</p>
                      <p className="card-subvalue">{(comparison.summary.highest_confidence.confidence * 100).toFixed(1)}%</p>
                    </div>
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Sentiment Distribution Comparison */}
          <div className="comparison-chart">
            <h3>📈 Sentiment Distribution Comparison</h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={prepareSentimentComparison()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="airline" />
                <YAxis />
                <Tooltip />
                <Legend />
                <Bar dataKey="positive" stackId="a" fill="#10b981" name="Positive" />
                <Bar dataKey="neutral" stackId="a" fill="#fbbf24" name="Neutral" />
                <Bar dataKey="negative" stackId="a" fill="#ef4444" name="Negative" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Percentage Comparison */}
          <div className="comparison-chart">
            <h3>📊 Sentiment Percentages & Ratings</h3>
            <ResponsiveContainer width="100%" height={400}>
              <BarChart data={preparePercentageComparison()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="airline" />
                <YAxis yAxisId="left" />
                <YAxis yAxisId="right" orientation="right" />
                <Tooltip />
                <Legend />
                <Bar yAxisId="left" dataKey="positive" fill="#10b981" name="Positive %" />
                <Bar yAxisId="left" dataKey="negative" fill="#ef4444" name="Negative %" />
                <Bar yAxisId="right" dataKey="rating" fill="#667eea" name="Avg Rating" />
                <Bar yAxisId="right" dataKey="confidence" fill="#fbbf24" name="Model Confidence %" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Rating Comparison */}
          <div className="comparison-chart">
            <h3>⭐ Average Rating Comparison</h3>
            <ResponsiveContainer width="100%" height={300}>
              <BarChart data={prepareRatingComparison()}>
                <CartesianGrid strokeDasharray="3 3" />
                <XAxis dataKey="airline" />
                <YAxis />
                <Tooltip />
                <Bar dataKey="rating" fill="#667eea" name="Average Rating" />
              </BarChart>
            </ResponsiveContainer>
          </div>

          {/* Top Issues Comparison */}
          {prepareIssuesComparison().length > 0 && (
            <div className="comparison-chart">
              <h3>🚨 Top Issues Comparison</h3>
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={prepareIssuesComparison()}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="issue" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  {Object.keys(comparison.comparison).filter(
                    key => !comparison.comparison[key].error
                  ).map((airline, index) => {
                    const label = availableAirlines.find(a => a.value === airline)?.label || airline;
                    return (
                      <Bar
                        key={airline}
                        dataKey={label}
                        fill={COLORS[index % COLORS.length]}
                        name={label}
                      />
                    );
                  })}
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {/* Detailed Stats Table */}
          <div className="detailed-stats">
            <h3>📋 Detailed Statistics</h3>
            <div className="stats-table-container">
              <table className="stats-table">
                <thead>
                  <tr>
                    <th>Airline</th>
                    <th>Total Reviews</th>
                    <th>Valid Reviews</th>
                    <th>Positive %</th>
                    <th>Negative %</th>
                    <th>Neutral %</th>
                    <th>Avg Rating</th>
                    <th>Model Confidence</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(comparison.comparison).map(([airline, data]) => {
                    if (data.error) return null;
                    const label = availableAirlines.find(a => a.value === airline)?.label || airline;
                    return (
                      <tr key={airline}>
                        <td><strong>{label}</strong></td>
                        <td>{data.total_reviews}</td>
                        <td>{data.valid_reviews}</td>
                        <td className="positive">{data.positive_percentage?.toFixed(1)}%</td>
                        <td className="negative">{data.negative_percentage?.toFixed(1)}%</td>
                        <td className="neutral">{data.neutral_percentage?.toFixed(1)}%</td>
                        <td>{data.average_rating?.toFixed(2)}</td>
                        <td>{(data.model_confidence * 100)?.toFixed(1)}%</td>
                      </tr>
                    );
                  })}
                </tbody>
              </table>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default CompareAirlines;

