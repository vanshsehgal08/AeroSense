import React, { useState, useEffect } from 'react';
import { sentimentAPI } from '../services/api';
import {
  BarChart,
  Bar,
  PieChart,
  Pie,
  Cell,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from 'recharts';
import './DataVisualization.css';

const DataVisualization = () => {
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
      setError(err.response?.data?.error || 'Failed to load data');
    } finally {
      setLoading(false);
    }
  };

  const COLORS = {
    Positive: '#10b981',
    Negative: '#ef4444',
    Neutral: '#f59e0b',
  };

  const getDataForChart = () => {
    if (!stats) return null;

    // Find the dataset with sentiment distribution
    for (const [key, data] of Object.entries(stats)) {
      if (data.sentiment_distribution) {
        return Object.entries(data.sentiment_distribution).map(([name, value]) => ({
          name,
          value,
        }));
      }
    }
    return null;
  };

  const getCategoryData = () => {
    if (!stats) return null;

    for (const [key, data] of Object.entries(stats)) {
      if (data.category_distribution) {
        return Object.entries(data.category_distribution)
          .sort(([, a], [, b]) => b - a)
          .slice(0, 10)
          .map(([name, value]) => ({
            name,
            value,
          }));
      }
    }
    return null;
  };

  const getRatingData = () => {
    if (!stats) return null;

    for (const [key, data] of Object.entries(stats)) {
      if (data.rating_distribution) {
        return Object.entries(data.rating_distribution)
          .sort(([a], [b]) => Number(a) - Number(b))
          .map(([name, value]) => ({
            name: `Rating ${name}`,
            value,
          }));
      }
    }
    return null;
  };

  if (loading) {
    return (
      <div className="data-visualization">
        <h2>Data Visualizations</h2>
        <div className="loading">Loading visualizations...</div>
      </div>
    );
  }

  if (error) {
    return (
      <div className="data-visualization">
        <h2>Data Visualizations</h2>
        <div className="error-message">{error}</div>
      </div>
    );
  }

  const sentimentData = getDataForChart();
  const categoryData = getCategoryData();
  const ratingData = getRatingData();

  return (
    <div className="data-visualization">
      <div className="viz-header">
        <h2>Data Visualizations</h2>
        <button className="refresh-button" onClick={loadStats}>
          🔄 Refresh
        </button>
      </div>

      <div className="charts-container">
        {sentimentData && (
          <div className="chart-section">
            <h3>Sentiment Distribution</h3>
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={400}>
                <PieChart>
                  <Pie
                    data={sentimentData}
                    cx="50%"
                    cy="50%"
                    labelLine={false}
                    label={({ name, percent }) =>
                      `${name}: ${(percent * 100).toFixed(0)}%`
                    }
                    outerRadius={120}
                    fill="#8884d8"
                    dataKey="value"
                  >
                    {sentimentData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={COLORS[entry.name] || '#8884d8'}
                      />
                    ))}
                  </Pie>
                  <Tooltip />
                  <Legend />
                </PieChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {sentimentData && (
          <div className="chart-section">
            <h3>Sentiment Counts</h3>
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={sentimentData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="value">
                    {sentimentData.map((entry, index) => (
                      <Cell
                        key={`cell-${index}`}
                        fill={COLORS[entry.name] || '#8884d8'}
                      />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {categoryData && (
          <div className="chart-section">
            <h3>Top Complaint Categories</h3>
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={categoryData} layout="vertical">
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis type="number" />
                  <YAxis dataKey="name" type="category" width={150} />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="value" fill="#3b82f6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {ratingData && (
          <div className="chart-section">
            <h3>Rating Distribution</h3>
            <div className="chart-wrapper">
              <ResponsiveContainer width="100%" height={400}>
                <BarChart data={ratingData}>
                  <CartesianGrid strokeDasharray="3 3" />
                  <XAxis dataKey="name" />
                  <YAxis />
                  <Tooltip />
                  <Legend />
                  <Bar dataKey="value" fill="#8b5cf6" />
                </BarChart>
              </ResponsiveContainer>
            </div>
          </div>
        )}

        {!sentimentData && !categoryData && !ratingData && (
          <div className="no-data">
            <p>No data available for visualization. Please upload data first.</p>
          </div>
        )}
      </div>
    </div>
  );
};

export default DataVisualization;

