import React, { useState, useEffect } from 'react';
import { sentimentAPI } from '../services/api';
import SentimentPredictor from './SentimentPredictor';
import DataUpload from './DataUpload';
import Statistics from './Statistics';
import ModelTraining from './ModelTraining';
import DataVisualization from './DataVisualization';
import ModelComparison from './ModelComparison';
import AdvancedNLP from './AdvancedNLP';
import AirlineInsights from './AirlineInsights';
import CompareAirlines from './CompareAirlines';
import './Dashboard.css';

const Dashboard = () => {
  const [activeTab, setActiveTab] = useState('predict');
  const [healthStatus, setHealthStatus] = useState(null);
  const [loading, setLoading] = useState(true);

  useEffect(() => {
    checkHealth();
  }, []);

  const checkHealth = async () => {
    try {
      const response = await sentimentAPI.healthCheck();
      setHealthStatus(response.data);
    } catch (error) {
      console.error('Health check failed:', error);
      setHealthStatus({ status: 'error', error: error.message });
    } finally {
      setLoading(false);
    }
  };

  const tabs = [
    { id: 'airline', label: 'Airline Insights', icon: '✈️' },
    { id: 'compare-airlines', label: 'Compare Airlines', icon: '🔀' },
    { id: 'upload', label: 'Upload', icon: '📤' },
    { id: 'train', label: 'Train', icon: '🎓' },
    { id: 'compare', label: 'Compare Models', icon: '⚖️' },
    { id: 'nlp', label: 'Advanced NLP', icon: '🧠' },
    { id: 'stats', label: 'Statistics', icon: '📊' },
    { id: 'visualize', label: 'Visualizations', icon: '📈' },
    { id: 'predict', label: 'Predict', icon: '🔮' },
  ];

  return (
    <div className="dashboard">
      <header className="dashboard-header">
        <h1>✈️ AeroSense - Sentiment Analysis Dashboard</h1>
        <div className="health-indicator">
          {loading ? (
            <span className="status-loading">Checking...</span>
          ) : healthStatus?.status === 'healthy' ? (
            <span className="status-healthy">
              ✓ Model Ready
              {healthStatus.model_trained && ` (Accuracy: ${(healthStatus.model_accuracy * 100).toFixed(2)}%)`}
            </span>
          ) : (
            <span className="status-error">✗ Service Unavailable</span>
          )}
        </div>
      </header>

      <nav className="dashboard-nav">
        {tabs.map((tab) => (
          <button
            key={tab.id}
            className={`nav-button ${activeTab === tab.id ? 'active' : ''}`}
            onClick={() => setActiveTab(tab.id)}
          >
            <span className="nav-icon">{tab.icon}</span>
            <span className="nav-label">{tab.label}</span>
          </button>
        ))}
      </nav>

      <main className="dashboard-content">
        {activeTab === 'predict' && <SentimentPredictor />}
        {activeTab === 'upload' && <DataUpload onUpload={checkHealth} />}
        {activeTab === 'stats' && <Statistics />}
        {activeTab === 'visualize' && <DataVisualization />}
        {activeTab === 'train' && <ModelTraining onTrain={checkHealth} />}
        {activeTab === 'compare' && <ModelComparison />}
        {activeTab === 'nlp' && <AdvancedNLP />}
        {activeTab === 'airline' && <AirlineInsights />}
        {activeTab === 'compare-airlines' && <CompareAirlines />}
      </main>
    </div>
  );
};

export default Dashboard;

