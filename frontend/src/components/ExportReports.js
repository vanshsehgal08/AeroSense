import React, { useState } from 'react';
import { sentimentAPI } from '../services/api';
import './ExportReports.css';

const ExportReports = () => {
  const [loading, setLoading] = useState(false);
  const [exported, setExported] = useState(null);
  const [error, setError] = useState(null);
  const [format, setFormat] = useState('excel');

  const handleExport = async () => {
    setLoading(true);
    setError(null);
    setExported(null);

    try {
      const response = await sentimentAPI.exportReport(format);
      setExported(response.data);
      
      // Trigger download if filename is provided
      if (response.data.filename) {
        downloadFile(response.data.filename);
      } else if (response.data.files) {
        // Multiple files (Excel)
        Object.values(response.data.files).forEach(filename => {
          downloadFile(filename);
        });
      }
    } catch (err) {
      setError(err.response?.data?.error || 'Export failed');
    } finally {
      setLoading(false);
    }
  };

  const downloadFile = async (filename) => {
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
      console.error('Download failed:', err);
    }
  };

  return (
    <div className="export-reports">
      <h2>Export Reports</h2>
      <p className="subtitle">Generate and download comprehensive analysis reports</p>

      <div className="export-container">
        <div className="export-options">
          <h3>Export Options</h3>
          
          <div className="format-selection">
            <label>
              <input
                type="radio"
                value="excel"
                checked={format === 'excel'}
                onChange={(e) => setFormat(e.target.value)}
              />
              Excel (.xlsx)
            </label>
            <label>
              <input
                type="radio"
                value="json"
                checked={format === 'json'}
                onChange={(e) => setFormat(e.target.value)}
              />
              JSON (.json)
            </label>
          </div>

          <div className="format-info">
            <h4>Export Includes:</h4>
            <ul>
              <li>Model information and accuracy</li>
              <li>Sentiment distribution statistics</li>
              <li>Category analysis</li>
              <li>Dataset summary</li>
              <li>Analysis metadata</li>
            </ul>
          </div>

          <button
            className="export-button"
            onClick={handleExport}
            disabled={loading}
          >
            {loading ? 'Generating Report...' : 'Export Report'}
          </button>
        </div>

        {error && (
          <div className="error-message">{error}</div>
        )}

        {exported && (
          <div className="export-success">
            <div className="success-message">
              ✓ Report exported successfully!
            </div>
            {exported.filename && (
              <div className="file-info">
                <p>File: {exported.filename}</p>
                <button
                  className="download-button"
                  onClick={() => downloadFile(exported.filename)}
                >
                  Download Again
                </button>
              </div>
            )}
            {exported.files && (
              <div className="files-info">
                <p>Files exported:</p>
                <ul>
                  {Object.entries(exported.files).map(([key, filename]) => (
                    <li key={key}>
                      {key}: {filename}
                      <button
                        className="download-link"
                        onClick={() => downloadFile(filename)}
                      >
                        Download
                      </button>
                    </li>
                  ))}
                </ul>
              </div>
            )}
          </div>
        )}

        <div className="export-instructions">
          <h3>Instructions</h3>
          <ul>
            <li><strong>Excel Format:</strong> Multi-sheet Excel file with comprehensive analysis</li>
            <li><strong>JSON Format:</strong> Structured JSON with all analysis data</li>
            <li>Reports include model information, statistics, and analysis results</li>
            <li>Files are automatically downloaded after generation</li>
          </ul>
        </div>
      </div>
    </div>
  );
};

export default ExportReports;

