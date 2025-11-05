import axios from 'axios';

const API_BASE_URL = process.env.REACT_APP_API_URL || (process.env.NODE_ENV === 'production' ? '/api' : 'http://localhost:5000/api');

const api = axios.create({
  baseURL: API_BASE_URL,
  headers: {
    'Content-Type': 'application/json',
  },
});

export const sentimentAPI = {
  // Health check
  healthCheck: () => api.get('/health'),

  // Predict sentiment for single review
  predictSentiment: (review) => api.post('/predict', { review }),

  // Predict sentiment for multiple reviews
  predictBatch: (reviews) => api.post('/predict-batch', { reviews }),

  // Upload and process CSV file
  uploadFile: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/upload', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  // Train model
  trainModel: (options = {}) => api.post('/train', options),

  // Get statistics
  getStats: () => api.get('/stats'),

  // Preprocess data
  preprocessData: (file) => {
    const formData = new FormData();
    formData.append('file', file);
    return api.post('/preprocess', formData, {
      headers: {
        'Content-Type': 'multipart/form-data',
      },
    });
  },

  // Download file
  downloadFile: (filename) => api.get(`/data/${filename}`, { responseType: 'blob' }),

  // Advanced ML Models
  compareModels: (options = {}) => api.post('/models/compare', options),
  tuneHyperparameters: (options = {}) => api.post('/models/tune', options),

  // Advanced NLP
  extractTopics: (options = {}) => api.post('/nlp/topics', options),
  extractKeywords: (options = {}) => api.post('/nlp/keywords', options),
  extractAspects: (text) => api.post('/nlp/aspects', { text }),
  extractEmotions: (text) => api.post('/nlp/emotions', { text }),

  // Model Interpretability
  explainPrediction: (text, topFeatures = 10) => api.post('/interpret/explain', { text, top_features: topFeatures }),
  getFeatureImportance: () => api.get('/interpret/features'),

  // Time Series
  getSentimentTrends: (options = {}) => api.post('/timeseries/trends', options),
  detectShifts: (options = {}) => api.post('/timeseries/shifts', options),

  // Export
  exportReport: (format = 'json') => api.post('/export/report', { format }),

  // Database
  getModelsFromDB: () => api.get('/database/models'),
  getPredictionsFromDB: (limit = 100) => api.get(`/database/predictions?limit=${limit}`),

  // Airline Insights
  scrapeAirlineReviews: (options = {}) => api.post('/airline/scrape', options),
  analyzeAirline: (options = {}) => api.post('/airline/analyze', options),
  scrapeAndAnalyze: (options = {}) => api.post('/airline/scrape-and-analyze', options),
};

export default api;

