import React, { useState, useEffect } from 'react';
import { sentimentAPI } from '../services/api';
import { BarChart, Bar, XAxis, YAxis, CartesianGrid, Tooltip, ResponsiveContainer } from 'recharts';
import './ModelInterpretability.css';

const ModelInterpretability = () => {
  const [activeTab, setActiveTab] = useState('explain');
  const [loading, setLoading] = useState(false);
  const [explanation, setExplanation] = useState(null);
  const [featureImportance, setFeatureImportance] = useState(null);
  const [error, setError] = useState(null);
  const [input, setInput] = useState('');

  useEffect(() => {
    loadFeatureImportance();
  }, []);

  const loadFeatureImportance = async () => {
    try {
      const response = await sentimentAPI.getFeatureImportance();
      setFeatureImportance(response.data);
    } catch (err) {
      console.error('Failed to load feature importance:', err);
    }
  };

  const handleExplain = async () => {
    if (!input.trim()) {
      setError('Please enter text to explain');
      return;
    }

    setLoading(true);
    setError(null);
    setExplanation(null);

    try {
      const response = await sentimentAPI.explainPrediction(input, 10);
      setExplanation(response.data);
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to explain prediction');
    } finally {
      setLoading(false);
    }
  };

  const prepareChartData = (features) => {
    if (!features) return [];
    return features.slice(0, 20).map((f, idx) => ({
      name: f.feature,
      importance: Math.abs(f.importance),
      positive: f.importance > 0,
    }));
  };

  return (
    <div className="model-interpretability">
      <h2>Model Interpretability</h2>
      <p className="subtitle">Understand how the model makes predictions</p>

      <div className="interpretability-container">
        <div className="interpret-tabs">
          <button
            className={`interpret-tab ${activeTab === 'explain' ? 'active' : ''}`}
            onClick={() => setActiveTab('explain')}
          >
            🔍 Explain Prediction
          </button>
          <button
            className={`interpret-tab ${activeTab === 'features' ? 'active' : ''}`}
            onClick={() => setActiveTab('features')}
          >
            📊 Feature Importance
          </button>
        </div>

        {activeTab === 'explain' && (
          <div className="explain-section">
            <div className="input-section">
              <textarea
                className="explain-input"
                placeholder="Enter text to explain..."
                value={input}
                onChange={(e) => setInput(e.target.value)}
                rows={6}
              />
              <button
                className="explain-button"
                onClick={handleExplain}
                disabled={loading || !input.trim()}
              >
                {loading ? 'Explaining...' : 'Explain Prediction'}
              </button>
            </div>

            {error && <div className="error-message">{error}</div>}

            {explanation && (
              <div className="explanation-results">
                <div className="prediction-summary">
                  <h4>Prediction</h4>
                  <div className="prediction-card">
                    <span className="prediction-label">Sentiment:</span>
                    <span className={`prediction-value ${explanation.prediction?.toLowerCase()}`}>
                      {explanation.prediction}
                    </span>
                  </div>
                  <div className="probabilities">
                    {Object.entries(explanation.probabilities || {}).map(([sentiment, prob]) => (
                      <div key={sentiment} className="probability-item">
                        <span>{sentiment}:</span>
                        <span>{(prob * 100).toFixed(2)}%</span>
                      </div>
                    ))}
                  </div>
                </div>

                {explanation.top_features && (
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

        {activeTab === 'features' && (
          <div className="features-section">
            {featureImportance?.feature_importance ? (
              <>
                <h4>Top 20 Most Important Features</h4>
                <ResponsiveContainer width="100%" height={500}>
                  <BarChart data={prepareChartData(featureImportance.feature_importance)} layout="vertical">
                    <CartesianGrid strokeDasharray="3 3" />
                    <XAxis type="number" />
                    <YAxis dataKey="name" type="category" width={150} />
                    <Tooltip />
                    <Bar dataKey="importance">
                      {prepareChartData(featureImportance.feature_importance).map((entry, index) => (
                        <Cell key={`cell-${index}`} fill={entry.positive ? '#10b981' : '#ef4444'} />
                      ))}
                    </Bar>
                  </BarChart>
                </ResponsiveContainer>

                <div className="features-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Feature</th>
                        <th>Importance</th>
                        {featureImportance.feature_importance[0]?.std && <th>Std Dev</th>}
                      </tr>
                    </thead>
                    <tbody>
                      {featureImportance.feature_importance.slice(0, 20).map((f, idx) => (
                        <tr key={idx}>
                          <td>{f.feature}</td>
                          <td>{f.importance.toFixed(6)}</td>
                          {f.std && <td>{f.std.toFixed(6)}</td>}
                        </tr>
                      ))}
                    </tbody>
                  </table>
                </div>
              </>
            ) : (
              <div className="loading">Loading feature importance...</div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelInterpretability;

