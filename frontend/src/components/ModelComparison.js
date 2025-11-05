import React, { useState } from 'react';
import { sentimentAPI } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer, Cell } from 'recharts';
import './ModelComparison.css';

const ModelComparison = () => {
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [options, setOptions] = useState({
    use_cleaned: true,
    cv_folds: 5,
    use_cache: true,
    force_retrain: false,
  });

  const handleCompare = async () => {
    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const response = await sentimentAPI.compareModels(options);
      setResults(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to compare models');
    } finally {
      setLoading(false);
    }
  };

  const prepareChartData = () => {
    if (!results || !results.comparison) return [];

    return Object.entries(results.comparison).map(([name, metrics]) => ({
      name: name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase()),
      accuracy: metrics.accuracy * 100,
      f1_score: metrics.f1_score * 100,
      precision: metrics.precision * 100,
      recall: metrics.recall * 100,
    }));
  };

  const chartData = prepareChartData();

  return (
    <div className="model-comparison">
      <h2>Model Comparison</h2>
      <p className="subtitle">Compare multiple ML algorithms and select the best one</p>

      <div className="comparison-container">
        <div className="comparison-options">
          <h3>Training Options</h3>
          
          <div className="option-group">
            <label>
              <input
                type="checkbox"
                checked={options.use_cleaned}
                onChange={(e) => setOptions({ ...options, use_cleaned: e.target.checked })}
              />
              Use cleaned reviews
            </label>
          </div>

          <div className="option-group">
            <label>
              Cross-Validation Folds: {options.cv_folds}
              <input
                type="range"
                min="3"
                max="10"
                value={options.cv_folds}
                onChange={(e) => setOptions({ ...options, cv_folds: parseInt(e.target.value) })}
              />
            </label>
          </div>

          <div className="option-group">
            <label>
              <input
                type="checkbox"
                checked={options.use_cache}
                onChange={(e) => setOptions({ ...options, use_cache: e.target.checked })}
              />
              Use cached models (faster)
            </label>
          </div>

          <div className="option-group">
            <label>
              <input
                type="checkbox"
                checked={options.force_retrain}
                onChange={(e) => setOptions({ ...options, force_retrain: e.target.checked })}
                disabled={!options.use_cache}
              />
              Force retrain (ignore cache)
            </label>
          </div>

          <button
            className="compare-button"
            onClick={handleCompare}
            disabled={loading}
          >
            {loading ? 'Comparing Models...' : 'Compare All Models'}
          </button>
        </div>

        {error && (
          <div className="error-message">{error}</div>
        )}

        {results && (
          <div className="comparison-results">
            <div className="results-header">
              <h3>Comparison Results</h3>
              <div className="results-badges">
                {results.from_cache && (
                  <div className="cache-badge">
                    ⚡ Loaded from Cache
                  </div>
                )}
                {results.best_model && (
                  <div className="best-model-badge">
                    🏆 Best Model: {results.best_model.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}
                  </div>
                )}
              </div>
            </div>
            
            {results.message && (
              <div className={`info-message ${results.from_cache ? 'cache-info' : 'training-info'}`}>
                {results.from_cache ? '⚡' : '🔄'} {results.message}
              </div>
            )}

            {chartData.length > 0 && (
              <div className="chart-section">
                <h4>Performance Metrics Comparison</h4>
                <ResponsiveContainer width="100%" height={400}>
                  <BarChart data={chartData}>
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis dataKey="name" angle={-45} textAnchor="end" height={100} />
                    <YAxis />
                    <Tooltip formatter={(value) => `${value.toFixed(2)}%`} />
                    <Legend />
                    <Bar dataKey="accuracy" fill="#8884d8" name="Accuracy %" />
                    <Bar dataKey="f1_score" fill="#82ca9d" name="F1-Score %" />
                    <Bar dataKey="precision" fill="#ffc658" name="Precision %" />
                    <Bar dataKey="recall" fill="#ff7300" name="Recall %" />
                  </BarChart>
                </ResponsiveContainer>
              </div>
            )}

            <div className="metrics-table">
              <h4>Detailed Metrics</h4>
              <table>
                <thead>
                  <tr>
                    <th>Model</th>
                    <th>Accuracy</th>
                    <th>F1-Score</th>
                    <th>Precision</th>
                    <th>Recall</th>
                    <th>CV Mean</th>
                    <th>CV Std</th>
                  </tr>
                </thead>
                <tbody>
                  {Object.entries(results.comparison).map(([name, metrics]) => (
                    <tr key={name} className={name === results.best_model ? 'best-model-row' : ''}>
                      <td>{name.replace('_', ' ').replace(/\b\w/g, l => l.toUpperCase())}</td>
                      <td>{(metrics.accuracy * 100).toFixed(2)}%</td>
                      <td>{(metrics.f1_score * 100).toFixed(2)}%</td>
                      <td>{(metrics.precision * 100).toFixed(2)}%</td>
                      <td>{(metrics.recall * 100).toFixed(2)}%</td>
                      <td>{(metrics.cv_mean * 100).toFixed(2)}%</td>
                      <td>{(metrics.cv_std * 100).toFixed(2)}%</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelComparison;

