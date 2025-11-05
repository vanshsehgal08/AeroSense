import React, { useState } from 'react';
import { sentimentAPI } from '../services/api';
import './DataUpload.css';

const DataUpload = ({ onUpload }) => {
  const [file, setFile] = useState(null);
  const [uploading, setUploading] = useState(false);
  const [result, setResult] = useState(null);
  const [error, setError] = useState(null);
  const [preprocessing, setPreprocessing] = useState(false);

  const handleFileChange = (e) => {
    setFile(e.target.files[0]);
    setResult(null);
    setError(null);
  };

  const handleUpload = async () => {
    if (!file) {
      setError('Please select a file');
      return;
    }

    setUploading(true);
    setError(null);

    try {
      const response = await sentimentAPI.uploadFile(file);
      setResult(response.data);
      if (onUpload) onUpload();
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to upload file');
    } finally {
      setUploading(false);
    }
  };

  const handlePreprocess = async () => {
    if (!file) {
      setError('Please select a file');
      return;
    }

    setPreprocessing(true);
    setError(null);

    try {
      const response = await sentimentAPI.preprocessData(file);
      setResult(response.data);
      if (onUpload) onUpload();
    } catch (err) {
      setError(err.response?.data?.error || 'Failed to preprocess file');
    } finally {
      setPreprocessing(false);
    }
  };

  const handleDownload = async (filename) => {
    try {
      const response = await sentimentAPI.downloadFile(filename);
      const url = window.URL.createObjectURL(new Blob([response.data]));
      const link = document.createElement('a');
      link.href = url;
      link.setAttribute('download', filename);
      document.body.appendChild(link);
      link.click();
      link.remove();
    } catch (err) {
      setError('Failed to download file');
    }
  };

  return (
    <div className="data-upload">
      <h2>Upload & Process Data</h2>
      <p className="subtitle">Upload CSV files for sentiment analysis</p>

      <div className="upload-container">
        <div className="upload-section">
          <div className="file-input-wrapper">
            <input
              type="file"
              id="file-input"
              accept=".csv"
              onChange={handleFileChange}
              className="file-input"
            />
            <label htmlFor="file-input" className="file-label">
              {file ? file.name : 'Choose CSV File'}
            </label>
          </div>

          <div className="upload-actions">
            <button
              className="action-button primary"
              onClick={handleUpload}
              disabled={!file || uploading || preprocessing}
            >
              {uploading ? 'Processing...' : 'Upload & Analyze'}
            </button>
            <button
              className="action-button secondary"
              onClick={handlePreprocess}
              disabled={!file || uploading || preprocessing}
            >
              {preprocessing ? 'Preprocessing...' : 'Preprocess Only'}
            </button>
          </div>

          {error && (
            <div className="error-message">{error}</div>
          )}

          {result && (
            <div className="result-section">
              <div className="success-message">
                ✓ {result.message || 'File processed successfully!'}
              </div>

              {result.stats && (
                <div className="stats-grid">
                  <div className="stat-card">
                    <div className="stat-value">{result.stats.total_reviews}</div>
                    <div className="stat-label">Total Reviews</div>
                  </div>

                  {result.stats.sentiment_distribution && (
                    <>
                      {Object.entries(result.stats.sentiment_distribution).map(([sentiment, count]) => (
                        <div key={sentiment} className="stat-card">
                          <div className="stat-value">{count}</div>
                          <div className="stat-label">{sentiment}</div>
                        </div>
                      ))}
                    </>
                  )}
                </div>
              )}

              {result.filename && (
                <button
                  className="download-button"
                  onClick={() => handleDownload(result.filename)}
                >
                  Download Processed File
                </button>
              )}
            </div>
          )}
        </div>

        <div className="instructions">
          <h3>Instructions</h3>
          <ul>
            <li>CSV file must contain a "Review" column</li>
            <li>Optional: Include "Rating" column for sentiment labeling</li>
            <li>Upload & Analyze: Process and predict sentiments</li>
            <li>Preprocess Only: Clean and prepare data for training</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default DataUpload;

