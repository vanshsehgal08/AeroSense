import React, { useState } from 'react';
import { sentimentAPI } from '../services/api';
import './AdvancedNLP.css';

const AdvancedNLP = () => {
  const [activeFeature, setActiveFeature] = useState('all');
  const [loading, setLoading] = useState(false);
  const [results, setResults] = useState(null);
  const [error, setError] = useState(null);
  const [input, setInput] = useState('');
  const [options, setOptions] = useState({
    n_topics: 5,
    n_keywords: 20,
    method: 'lda',
  });

  const features = [
    { id: 'all', label: 'All Features', icon: '🚀' },
    { id: 'topics', label: 'Topic Modeling', icon: '📚' },
    { id: 'keywords', label: 'Keyword Extraction', icon: '🔑' },
    { id: 'aspects', label: 'Aspect Extraction', icon: '🎯' },
    { id: 'emotions', label: 'Emotion Analysis', icon: '😊' },
  ];

  const handleAnalyze = async () => {
    if (!input.trim()) {
      setError('Please enter text or upload data');
      return;
    }

    setLoading(true);
    setError(null);
    setResults(null);

    try {
      const allResults = {};
      const texts = input.split('\n').filter(t => t.trim());
      const processedTexts = texts.length === 1 ? [texts[0], texts[0]] : texts;
      const fullText = input.trim();

      // Run all analyses in parallel
      const promises = [];

      // Topics
      if (activeFeature === 'all' || activeFeature === 'topics') {
        promises.push(
          sentimentAPI.extractTopics({
            texts: processedTexts,
            n_topics: options.n_topics,
            method: options.method,
          }).then(res => ({ type: 'topics', data: res.data })).catch(err => ({ type: 'topics', error: err.message }))
        );
      }

      // Keywords
      if (activeFeature === 'all' || activeFeature === 'keywords') {
        promises.push(
          sentimentAPI.extractKeywords({
            texts: processedTexts,
            n_keywords: options.n_keywords,
          }).then(res => ({ type: 'keywords', data: res.data })).catch(err => ({ type: 'keywords', error: err.message }))
        );
      }

      // Aspects
      if (activeFeature === 'all' || activeFeature === 'aspects') {
        promises.push(
          sentimentAPI.extractAspects(fullText)
            .then(res => ({ type: 'aspects', data: res.data }))
            .catch(err => ({ type: 'aspects', error: err.message }))
        );
      }

      // Emotions
      if (activeFeature === 'all' || activeFeature === 'emotions') {
        promises.push(
          sentimentAPI.extractEmotions(fullText)
            .then(res => ({ type: 'emotions', data: res.data }))
            .catch(err => ({ type: 'emotions', error: err.message }))
        );
      }

      // Wait for all promises
      const responses = await Promise.all(promises);

      // Combine all results
      responses.forEach(({ type, data, error }) => {
        if (error) {
          allResults[type] = { error };
        } else {
          allResults[type] = data;
        }
      });

      setResults(allResults);
    } catch (err) {
      console.error('Analysis error:', err);
      const errorMsg = err.response?.data?.error || err.message || 'Analysis failed';
      setError(errorMsg);
      setResults(null);
    } finally {
      setLoading(false);
    }
  };

  const renderResults = () => {
    if (!results) return null;

    // If showing all features, display all results
    if (activeFeature === 'all') {
      return (
        <div className="all-results">
          {/* Topics */}
          {results.topics && (
            <div className="results-section">
              <h3>📚 Topic Modeling</h3>
              <div className="topics-results">
                {results.topics.error ? (
                  <div className="error-text">{results.topics.error}</div>
                ) : (
                  (Array.isArray(results.topics) ? results.topics : (results.topics.topics || [])).map((topic, idx) => (
                    <div key={idx} className="topic-card">
                      <h5>Topic {topic.topic_id + 1}</h5>
                      <div className="topic-words">
                        {topic.words?.map((word, i) => (
                          <span key={i} className="topic-word">
                            {word} ({topic.scores?.[i]?.toFixed(3) || '0.000'})
                          </span>
                        ))}
                      </div>
                    </div>
                  ))
                )}
              </div>
            </div>
          )}

          {/* Keywords */}
          {results.keywords && (
            <div className="results-section">
              <h3>🔑 Keyword Extraction</h3>
              <div className="keywords-results">
                {results.keywords.error ? (
                  <div className="error-text">{results.keywords.error}</div>
                ) : (
                  <div className="keywords-list">
                    {(Array.isArray(results.keywords) ? results.keywords : (results.keywords.keywords || [])).map((kw, idx) => (
                      <div key={idx} className="keyword-item">
                        <span className="keyword-word">{kw.word}</span>
                        <span className="keyword-score">{kw.score?.toFixed(4) || kw}</span>
                      </div>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Aspects */}
          {results.aspects && (
            <div className="results-section">
              <h3>🎯 Aspect Extraction</h3>
              <div className="aspects-results">
                {results.aspects.error ? (
                  <div className="error-text">{results.aspects.error}</div>
                ) : (
                  <div className="aspects-list">
                    {(Array.isArray(results.aspects) ? results.aspects : (results.aspects.aspects || [])).map((aspect, idx) => (
                      <span key={idx} className="aspect-badge">{aspect}</span>
                    ))}
                  </div>
                )}
              </div>
            </div>
          )}

          {/* Emotions */}
          {results.emotions && (
            <div className="results-section">
              <h3>😊 Emotion Analysis</h3>
              <div className="emotions-results">
                <div className="dominant-emotion">
                  Dominant Emotion: <strong>{results.emotions.dominant_emotion}</strong>
                </div>
                <div className="emotions-grid">
                  {Object.entries(results.emotions.emotions || {}).map(([emotion, score]) => (
                    <div key={emotion} className="emotion-item">
                      <div className="emotion-label">{emotion}</div>
                      <div className="emotion-bar">
                        <div
                          className="emotion-fill"
                          style={{ width: `${score * 100}%` }}
                        />
                      </div>
                      <div className="emotion-score">{(score * 100).toFixed(1)}%</div>
                    </div>
                  ))}
                </div>
                {results.emotions.error && <div className="error-text">{results.emotions.error}</div>}
              </div>
            </div>
          )}
        </div>
      );
    }

    // Single feature mode (backward compatibility)
    switch (activeFeature) {
      case 'topics': {
        // Handle both array format and object with topics property
        const topicsData = Array.isArray(results.topics)
          ? results.topics
          : (results.topics?.topics || []);
        
        return (
          <div className="topics-results">
            <h4>Extracted Topics</h4>
            {results.topics?.error ? (
              <div className="error-text">{results.topics.error}</div>
            ) : (
              topicsData.map((topic, idx) => (
                <div key={idx} className="topic-card">
                  <h5>Topic {topic.topic_id + 1}</h5>
                  <div className="topic-words">
                    {topic.words?.map((word, i) => (
                      <span key={i} className="topic-word">
                        {word} ({topic.scores?.[i]?.toFixed(3) || '0.000'})
                      </span>
                    ))}
                  </div>
                </div>
              ))
            )}
          </div>
        );
      }

      case 'keywords': {
        // Handle both array format and object with keywords property
        const keywordsData = Array.isArray(results.keywords) 
          ? results.keywords 
          : (results.keywords?.keywords || []);
        
        return (
          <div className="keywords-results">
            <h4>Top Keywords</h4>
            {results.keywords?.error ? (
              <div className="error-text">{results.keywords.error}</div>
            ) : (
              <div className="keywords-list">
                {keywordsData.map((kw, idx) => (
                  <div key={idx} className="keyword-item">
                    <span className="keyword-word">{kw.word}</span>
                    <span className="keyword-score">{kw.score?.toFixed(4) || kw}</span>
                  </div>
                ))}
              </div>
            )}
          </div>
        );
      }

      case 'aspects': {
        // Handle both array format and object with aspects property
        const aspectsData = Array.isArray(results.aspects)
          ? results.aspects
          : (results.aspects?.aspects || []);
        
        return (
          <div className="aspects-results">
            <h4>Extracted Aspects</h4>
            {results.aspects?.error ? (
              <div className="error-text">{results.aspects.error}</div>
            ) : (
              <div className="aspects-list">
                {aspectsData.map((aspect, idx) => (
                  <span key={idx} className="aspect-badge">{aspect}</span>
                ))}
              </div>
            )}
          </div>
        );
      }

      case 'emotions': {
        // Handle object format with emotions property
        const emotionsData = results.emotions?.emotions || results.emotions || {};
        const dominantEmotion = results.emotions?.dominant_emotion || results.dominant_emotion;
        
        return (
          <div className="emotions-results">
            <h4>Emotion Analysis</h4>
            {results.emotions?.error ? (
              <div className="error-text">{results.emotions.error}</div>
            ) : (
              <>
                <div className="dominant-emotion">
                  Dominant Emotion: <strong>{dominantEmotion || 'N/A'}</strong>
                </div>
                <div className="emotions-grid">
                  {Object.entries(emotionsData).map(([emotion, score]) => (
                    <div key={emotion} className="emotion-item">
                      <div className="emotion-label">{emotion}</div>
                      <div className="emotion-bar">
                        <div
                          className="emotion-fill"
                          style={{ width: `${(score * 100) || 0}%` }}
                        />
                      </div>
                      <div className="emotion-score">{((score || 0) * 100).toFixed(1)}%</div>
                    </div>
                  ))}
                </div>
              </>
            )}
          </div>
        );
      }

      default:
        return null;
    }
  };

  return (
    <div className="advanced-nlp">
      <h2>Advanced NLP Features</h2>
      <p className="subtitle">Extract topics, keywords, aspects, and emotions from text</p>

      <div className="nlp-container">
        <div className="feature-tabs">
          {features.map((feature) => (
            <button
              key={feature.id}
              className={`feature-tab ${activeFeature === feature.id ? 'active' : ''}`}
              onClick={() => {
                setActiveFeature(feature.id);
                setResults(null);
              }}
            >
              <span className="tab-icon">{feature.icon}</span>
              <span className="tab-label">{feature.label}</span>
            </button>
          ))}
        </div>

        <div className="nlp-content">
          <div className="input-section">
            <textarea
              className="nlp-input"
              placeholder={
                activeFeature === 'aspects' || activeFeature === 'emotions'
                  ? 'Enter text to analyze... (e.g., "This product is amazing! Great quality and fast delivery.")'
                  : 'Enter multiple texts, one per line... (e.g., "This product is great" on line 1, "I love this service" on line 2, etc.)'
              }
              value={input}
              onChange={(e) => setInput(e.target.value)}
              rows={8}
            />
            {(activeFeature === 'all' || activeFeature === 'topics' || activeFeature === 'keywords') && (
              <div className="info-message">
                💡 Tip: Enter multiple texts (one per line) for better results. For single text, it will be duplicated automatically.
              </div>
            )}

            {(activeFeature === 'all' || activeFeature === 'topics' || activeFeature === 'keywords') && (
              <div className="options-section">
                {(activeFeature === 'all' || activeFeature === 'topics') && (
                  <>
                    <label>
                      Number of Topics:
                      <input
                        type="number"
                        min="2"
                        max="20"
                        value={options.n_topics}
                        onChange={(e) => setOptions({ ...options, n_topics: parseInt(e.target.value) })}
                      />
                    </label>
                    <label>
                      Method:
                      <select
                        value={options.method}
                        onChange={(e) => setOptions({ ...options, method: e.target.value })}
                      >
                        <option value="lda">LDA</option>
                        <option value="nmf">NMF</option>
                      </select>
                    </label>
                  </>
                )}
                {(activeFeature === 'all' || activeFeature === 'keywords') && (
                  <label>
                    Number of Keywords:
                    <input
                      type="number"
                      min="5"
                      max="50"
                      value={options.n_keywords}
                      onChange={(e) => setOptions({ ...options, n_keywords: parseInt(e.target.value) })}
                    />
                  </label>
                )}
              </div>
            )}

            <button
              className="analyze-button"
              onClick={handleAnalyze}
              disabled={loading || !input.trim()}
            >
              {loading 
                ? (activeFeature === 'all' ? 'Analyzing All Features...' : 'Analyzing...') 
                : (activeFeature === 'all' ? '🚀 Analyze All Features' : 'Analyze')}
            </button>
          </div>

          {error && <div className="error-message">{error}</div>}

          {results && renderResults()}
        </div>
      </div>
    </div>
  );
};

export default AdvancedNLP;

