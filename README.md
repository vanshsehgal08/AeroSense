# AeroSense

A production-ready, industry-grade sentiment analysis platform with advanced ML capabilities and explainable AI features.

## 🎯 Features

- **Multiple ML Models**: Compare 6 algorithms (Logistic Regression, Random Forest, SVM, Naive Bayes, Gradient Boosting, Neural Network) side-by-side
- **Advanced NLP**: Topic modeling (LDA & NMF), emotion analysis, keyword extraction, aspect extraction
- **Explainable AI**: Model interpretability with feature importance and prediction explanations
- **Time Series Analysis**: Sentiment trend detection, shift detection, and forecasting
- **Real-time Predictions**: Instant sentiment analysis with confidence scores
- **Batch Processing**: Analyze large datasets efficiently
- **Model Versioning**: Track model history with database persistence
- **Export & Reporting**: Multi-format exports (Excel, CSV, JSON) with comprehensive reports
- **Interactive Dashboard**: Modern React UI with real-time visualizations

## 🤖 Model & Tech Stack

### Machine Learning Models
- **6 ML Algorithms** with automatic comparison and hyperparameter tuning
- **TF-IDF Vectorization** for feature extraction
- **Cross-validation** and ensemble methods for robust performance
- **Model Caching** for faster predictions

**ML Models Used:**

1. **Logistic Regression** - Linear classification model that predicts probabilities using a sigmoid function. Fast, interpretable, and works well as a baseline for text classification tasks. Excellent for understanding feature importance through coefficients.

2. **Random Forest** - Ensemble method that combines multiple decision trees for robust predictions. Handles non-linear relationships well and is less prone to overfitting. Provides feature importance rankings naturally through tree-based splits.

3. **Support Vector Machine (SVM)** - Finds optimal boundaries between classes using kernel functions. Particularly effective for high-dimensional text data. Linear kernel works well for sparse TF-IDF vectors and provides good separation margins.

4. **Naive Bayes** - Probabilistic classifier based on Bayes' theorem with independence assumptions. Extremely fast for training and prediction, making it ideal for large datasets. Works well even with limited training data.

5. **Gradient Boosting** - Sequential ensemble method that builds models to correct previous errors. High predictive accuracy through iterative optimization. Often achieves best performance by combining weak learners into a strong model.

6. **Neural Network (MLP)** - Multi-layer perceptron with hidden layers for complex pattern recognition. Captures non-linear relationships in text data through backpropagation. Deep learning approach that can learn intricate feature combinations.

### Topic Modeling Methods

- **LDA (Latent Dirichlet Allocation)** - Probabilistic model that discovers hidden topics in text collections. Assumes documents are mixtures of topics and topics are distributions of words. Excellent for uncovering thematic structure and understanding main themes across large document sets.

- **NMF (Non-negative Matrix Factorization)** - Matrix factorization technique that decomposes documents into topics and word distributions. Produces additive, interpretable topic representations. Often provides clearer, more distinct topics compared to LDA, especially for specific domains.

### Tech Stack
**Backend:**
- Flask (RESTful API)
- scikit-learn (ML algorithms)
- spaCy & NLTK (Advanced NLP)
- pandas, numpy (Data processing)
- SQLite (Database versioning)

**Frontend:**
- React (Modern UI)
- Recharts (Visualizations)

## ✨ Why It's Unique & Helpful

**Not Just Another Sentiment Analyzer** - This platform goes beyond basic sentiment classification:

1. **Model Comparison**: Automatically tests 6 different algorithms to find the best model for your data
2. **Explainable AI**: Understand *why* predictions are made with feature importance and explanations
3. **Advanced Insights**: Discover topics, emotions, and patterns in your text data, not just sentiment
4. **Time Intelligence**: Track sentiment trends over time and detect significant shifts
5. **Production Ready**: Complete with model versioning, database persistence, and comprehensive reporting
6. **All-in-One**: From raw text to actionable insights in one platform

Perfect for businesses analyzing customer reviews, feedback, or any text data where understanding sentiment and underlying themes matters.
