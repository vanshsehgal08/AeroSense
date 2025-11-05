import React, { useState } from 'react';
import { sentimentAPI } from '../services/api';
import './ModelTraining.css';

const ModelTraining = ({ onTrain }) => {
  const [training, setTraining] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [options, setOptions] = useState({
    use_cleaned: true,
    test_size: 0.2,
  });

  const handleTrain = async () => {
    setTraining(true);
    setError(null);
    setResult(null);

    try {
      const response = await sentimentAPI.trainModel(options);
      setResult(response.data);
      if (onTrain) onTrain();
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to train model');
    } finally {
      setTraining(false);
    }
  };

  return (
    <div className="model-training">
      <h2>Train Model</h2>
      <p className="subtitle">Train or retrain the sentiment analysis model</p>

      <div className="training-container">
        <div className="training-options">
          <h3>Training Options</h3>
          
          <div className="option-group">
            <label>
              <input
                type="checkbox"
                checked={options.use_cleaned}
                onChange={(e) =>
                  setOptions({ ...options, use_cleaned: e.target.checked })
                }
              />
              Use cleaned reviews (recommended)
            </label>
          </div>

          <div className="option-group">
            <label>
              Test Size: {options.test_size}
              <input
                type="range"
                min="0.1"
                max="0.5"
                step="0.05"
                value={options.test_size}
                onChange={(e) =>
                  setOptions({ ...options, test_size: parseFloat(e.target.value) })
                }
              />
            </label>
          </div>

          <button
            className="train-button"
            onClick={handleTrain}
            disabled={training}
          >
            {training ? 'Training Model...' : 'Train Model'}
          </button>
        </div>

        {error && (
          <div className="error-message">{error}</div>
        )}

        {result && (
          <div className="training-results">
            <div className="success-message">
              ✓ Model trained successfully!
            </div>

            <div className="results-grid">
              <div className="result-card">
                <div className="result-value">
                  {(result.accuracy * 100).toFixed(2)}%
                </div>
                <div className="result-label">Accuracy</div>
              </div>

              {result.model_info && (
                <>
                  <div className="result-card">
                    <div className="result-value">
                      {result.model_info.training_samples}
                    </div>
                    <div className="result-label">Training Samples</div>
                  </div>
                  <div className="result-card">
                    <div className="result-value">
                      {result.model_info.test_samples}
                    </div>
                    <div className="result-label">Test Samples</div>
                  </div>
                </>
              )}
            </div>

            {result.classification_report && (
              <div className="classification-report">
                <h4>Classification Report</h4>
                <div className="report-table">
                  <table>
                    <thead>
                      <tr>
                        <th>Class</th>
                        <th>Precision</th>
                        <th>Recall</th>
                        <th>F1-Score</th>
                        <th>Support</th>
                      </tr>
                    </thead>
                    <tbody>
                      {Object.entries(result.classification_report)
                        .filter(([key]) => !['accuracy', 'macro avg', 'weighted avg'].includes(key))
                        .map(([className, metrics]) => (
                          <tr key={className}>
                            <td>{className}</td>
                            <td>{metrics.precision?.toFixed(3)}</td>
                            <td>{metrics.recall?.toFixed(3)}</td>
                            <td>{metrics['f1-score']?.toFixed(3)}</td>
                            <td>{metrics.support}</td>
                          </tr>
                        ))}
                    </tbody>
                  </table>
                </div>
              </div>
            )}
          </div>
        )}
      </div>
    </div>
  );
};

export default ModelTraining;

