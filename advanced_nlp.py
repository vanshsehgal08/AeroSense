"""
Advanced NLP Features
Includes BERT embeddings, topic modeling, NER, and aspect extraction
"""

import numpy as np
import pandas as pd
from sklearn.decomposition import LatentDirichletAllocation, NMF
from sklearn.feature_extraction.text import CountVectorizer
import spacy
from collections import Counter
import re

class AdvancedNLP:
    """Advanced NLP processing and analysis"""
    
    def __init__(self):
        self.nlp = None
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            print("spaCy model not loaded. Some features may not work.")
    
    def extract_aspects(self, text):
        """Extract aspects/entities from text using spaCy"""
        if not text or not isinstance(text, str) or not text.strip():
            return []
        
        # Common stop words and non-meaningful words to exclude
        stop_words_extended = {
            'were', 'have', 'has', 'had', 'been', 'being', 'will', 'would', 'could',
            'should', 'may', 'might', 'can', 'cannot', 'this', 'that', 'these', 'those',
            'the', 'a', 'an', 'and', 'or', 'but', 'if', 'when', 'where', 'why', 'how',
            'what', 'which', 'who', 'whom', 'whose', 'there', 'here', 'then', 'than',
            'more', 'most', 'less', 'least', 'some', 'any', 'all', 'each', 'every',
            'both', 'either', 'neither', 'one', 'two', 'first', 'second', 'last',
            'very', 'quite', 'rather', 'too', 'also', 'just', 'only', 'even', 'still',
            'yet', 'already', 'again', 'always', 'never', 'often', 'sometimes', 'usually'
        }
        
        if not self.nlp:
            # Fallback: extract simple noun phrases using regex
            from nltk.tokenize import word_tokenize
            from nltk.corpus import stopwords
            
            try:
                stop_words = set(stopwords.words('english')) | stop_words_extended
                words = word_tokenize(text.lower())
                
                aspects = []
                # Look for noun phrases (adj + noun patterns)
                for i in range(len(words) - 1):
                    if (words[i] not in stop_words and words[i].isalpha() and len(words[i]) > 3 and
                        words[i+1] not in stop_words and words[i+1].isalpha() and len(words[i+1]) > 3):
                        phrase = f"{words[i]} {words[i+1]}"
                        aspects.append(phrase)
                
                # Also add significant single words
                for word in words:
                    if (word not in stop_words and word.isalpha() and 
                        len(word) > 4 and word not in ['with', 'from', 'into', 'onto']):
                        aspects.append(word)
                
                return list(set(aspects[:20]))  # Limit to 20
            except:
                return []
        
        try:
            doc = self.nlp(text)
            aspects = []
            stop_words = set(stopwords.words('english')) | stop_words_extended
            
            # Extract noun phrases (filter out stop words)
            for chunk in doc.noun_chunks:
                # Check if chunk contains meaningful words
                chunk_text = chunk.text.lower().strip()
                chunk_words = chunk_text.split()
                
                # Filter out chunks that are mostly stop words
                meaningful_words = [w for w in chunk_words if w not in stop_words and len(w) > 2]
                
                if (len(meaningful_words) >= 1 and  # At least one meaningful word
                    len(chunk_text.split()) <= 3 and  # Max 3 words
                    len(chunk_text) > 3):  # Min length
                    # Further filter: remove if it's just common words
                    if not all(word in stop_words_extended for word in chunk_words):
                        aspects.append(chunk_text)
            
            # Extract named entities (only meaningful ones)
            for ent in doc.ents:
                if (ent.label_ in ['ORG', 'PRODUCT', 'PERSON', 'EVENT', 'GPE', 'MONEY'] and
                    len(ent.text.strip()) > 2 and
                    ent.text.lower() not in stop_words_extended):
                    aspects.append(ent.text.lower().strip())
            
            # Remove duplicates and filter
            unique_aspects = []
            seen = set()
            for aspect in aspects:
                aspect_lower = aspect.lower().strip()
                # Skip if it's a stop word or too short
                if (aspect_lower not in seen and 
                    aspect_lower not in stop_words_extended and
                    len(aspect_lower) > 3 and
                    not all(char in [' ', 'a', 'e', 'i', 'o', 'u'] for char in aspect_lower.replace(' ', ''))):
                    seen.add(aspect_lower)
                    unique_aspects.append(aspect_lower)
            
            return unique_aspects[:30]  # Limit to 30 most relevant
        except Exception as e:
            print(f"Aspect extraction error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def extract_keywords(self, texts, n_keywords=20, min_df=1):
        """Extract most important keywords using TF-IDF with lemmatization"""
        from sklearn.feature_extraction.text import TfidfVectorizer
        import re
        from nltk.tokenize import word_tokenize
        from nltk.corpus import stopwords
        
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Filter out empty texts
        texts = [t for t in texts if t and t.strip()]
        
        if not texts:
            return []
        
        # Adjust min_df based on number of texts
        if len(texts) < min_df:
            min_df = 1
        
        try:
            # Lemmatize texts before vectorization
            lemmatized_texts = []
            stop_words = set(stopwords.words('english'))
            
            for text in texts:
                # Preprocess: lowercase, tokenize, lemmatize
                if self.nlp:
                    doc = self.nlp(text.lower())
                    # Get lemmatized tokens, filter stop words and short words
                    lemmas = [token.lemma_ for token in doc 
                             if not token.is_stop 
                             and not token.is_punct 
                             and len(token.lemma_) > 2 
                             and token.lemma_.isalpha()]
                    lemmatized_text = ' '.join(lemmas)
                else:
                    # Fallback: simple tokenization
                    tokens = word_tokenize(text.lower())
                    lemmas = [t for t in tokens if t not in stop_words and len(t) > 2 and t.isalpha()]
                    lemmatized_text = ' '.join(lemmas)
                
                lemmatized_texts.append(lemmatized_text)
            
            # Use lemmatized texts for TF-IDF
            vectorizer = TfidfVectorizer(
                max_features=n_keywords * 3,  # Get more to filter duplicates
                min_df=min_df, 
                stop_words='english',
                token_pattern=r'\b[a-z]{3,}\b'  # Only words with 3+ letters
            )
            tfidf_matrix = vectorizer.fit_transform(lemmatized_texts)
            feature_names = vectorizer.get_feature_names_out()
            
            if len(feature_names) == 0:
                return []
            
            # Calculate TF-IDF scores properly - sum across all documents, then normalize
            # This gives us the importance of each word across all texts
            tfidf_array = tfidf_matrix.toarray()
            
            # Sum TF-IDF scores across all documents for each word
            word_scores = np.sum(tfidf_array, axis=0)
            
            # Normalize by document count to get average, but also consider max
            # This helps differentiate words that appear in multiple documents
            doc_freq = np.sum(tfidf_array > 0, axis=0)  # How many documents contain each word
            normalized_scores = word_scores * (1 + doc_freq / len(lemmatized_texts))
            
            keywords = list(zip(feature_names, normalized_scores))
            keywords.sort(key=lambda x: x[1], reverse=True)
            
            # Remove duplicates and short words, limit to n_keywords
            seen = set()
            unique_keywords = []
            for word, score in keywords:
                # Check for similar words (basic deduplication)
                word_lower = word.lower()
                if word_lower not in seen and len(word) > 2:
                    seen.add(word_lower)
                    # Round to 4 decimal places for better display
                    unique_keywords.append({
                        'word': word, 
                        'score': round(float(score), 4)
                    })
                    if len(unique_keywords) >= n_keywords:
                        break
            
            return unique_keywords
        except Exception as e:
            print(f"Keyword extraction error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def topic_modeling(self, texts, n_topics=5, method='lda'):
        """Perform topic modeling using LDA or NMF"""
        # Ensure texts is a list
        if isinstance(texts, str):
            texts = [texts]
        
        # Filter out empty texts
        texts = [t for t in texts if t and t.strip()]
        
        if not texts:
            return []
        
        # Adjust n_topics if we have fewer texts
        if len(texts) < n_topics:
            n_topics = max(2, len(texts) - 1)
        
        # Adjust min_df based on number of texts
        min_df = max(1, min(2, len(texts) // 10))
        
        try:
            vectorizer = CountVectorizer(max_features=100, min_df=min_df, stop_words='english')
            doc_term_matrix = vectorizer.fit_transform(texts)
            feature_names = vectorizer.get_feature_names_out()
            
            if len(feature_names) == 0:
                return []
            
            if method == 'lda':
                model = LatentDirichletAllocation(n_components=n_topics, random_state=42, max_iter=10)
            else:  # NMF
                from sklearn.decomposition import NMF
                model = NMF(n_components=n_topics, random_state=42, max_iter=200)
            
            model.fit(doc_term_matrix)
            
            topics = []
            for idx, topic in enumerate(model.components_):
                # Get top words for this topic
                top_words_count = min(10, len(feature_names))
                top_words_idx = topic.argsort()[-top_words_count:][::-1]
                top_words = [feature_names[i] for i in top_words_idx]
                top_scores = [float(topic[i]) for i in top_words_idx]
                
                topics.append({
                    'topic_id': idx,
                    'words': top_words,
                    'scores': top_scores
                })
            
            return topics
        except Exception as e:
            print(f"Topic modeling error: {e}")
            import traceback
            traceback.print_exc()
            return []
    
    def extract_entities(self, text):
        """Extract named entities using spaCy"""
        if not self.nlp:
            return []
        
        doc = self.nlp(text)
        entities = []
        
        for ent in doc.ents:
            entities.append({
                'text': ent.text,
                'label': ent.label_,
                'description': spacy.explain(ent.label_)
            })
        
        return entities
    
    def analyze_sentiment_intensity(self, text):
        """Analyze sentiment intensity (not just polarity)"""
        if not self.nlp:
            return {'intensity': 0.5, 'polarity': 'neutral'}
        
        # Simple intensity calculation based on emotional words
        positive_words = ['excellent', 'amazing', 'wonderful', 'great', 'fantastic', 'love', 'perfect']
        negative_words = ['terrible', 'awful', 'horrible', 'bad', 'hate', 'worst', 'disgusting']
        
        text_lower = text.lower()
        positive_count = sum(1 for word in positive_words if word in text_lower)
        negative_count = sum(1 for word in negative_words if word in text_lower)
        
        # Calculate intensity
        intensity = min(abs(positive_count - negative_count) / 10, 1.0)
        
        if positive_count > negative_count:
            polarity = 'positive'
        elif negative_count > positive_count:
            polarity = 'negative'
        else:
            polarity = 'neutral'
        
        return {
            'intensity': float(intensity),
            'polarity': polarity,
            'positive_words': positive_count,
            'negative_words': negative_count
        }
    
    def extract_emotions(self, text):
        """Extract emotional content with improved keyword matching"""
        if not text or not isinstance(text, str) or not text.strip():
            return {
                'emotions': {'joy': 0.0, 'anger': 0.0, 'sadness': 0.0, 'fear': 0.0, 'surprise': 0.0, 'disgust': 0.0},
                'dominant_emotion': None
            }
        
        # Expanded emotion keywords with variations
        emotions = {
            'joy': [
                'happy', 'joyful', 'excited', 'delighted', 'pleased', 'glad', 'great', 'wonderful', 
                'fantastic', 'excellent', 'amazing', 'awesome', 'love', 'enjoy', 'enjoyed', 'enjoying',
                'satisfied', 'satisfaction', 'pleasure', 'pleased', 'content', 'thrilled', 'ecstatic',
                'bliss', 'cheerful', 'jubilant', 'merry', 'upbeat', 'positive', 'good', 'nice', 'fine',
                'perfect', 'best', 'favorite', 'favourite', 'brilliant', 'outstanding', 'superb'
            ],
            'anger': [
                'angry', 'furious', 'mad', 'annoyed', 'irritated', 'frustrated', 'rage', 'raging',
                'outraged', 'livid', 'incensed', 'enraged', 'infuriated', 'irritating', 'frustrating',
                'annoying', 'hate', 'hated', 'hates', 'disgusting', 'horrible', 'awful', 'terrible',
                'worst', 'bad', 'worst', 'disappointed', 'disappointing', 'unacceptable', 'unfair',
                'unprofessional', 'rude', 'rudeness', 'incompetent', 'useless', 'pathetic', 'poor',
                'poorly', 'failed', 'failure', 'failing', 'wrong', 'mistake', 'errors', 'problem',
                'problems', 'issues', 'issue', 'complaint', 'complain', 'complained', 'outrage'
            ],
            'sadness': [
                'sad', 'depressed', 'unhappy', 'disappointed', 'upset', 'down', 'miserable', 'unfortunate',
                'unfortunately', 'regret', 'regrettable', 'sorry', 'sorrow', 'grief', 'grieving',
                'heartbroken', 'devastated', 'devastating', 'tragic', 'tragedy', 'unfortunate',
                'disappointment', 'disappointing', 'let down', 'letdown', 'dissatisfied', 'dissatisfaction',
                'unpleasant', 'unpleasantly', 'uncomfortable', 'uncomfortably', 'hurt', 'hurtful'
            ],
            'fear': [
                'afraid', 'scared', 'worried', 'anxious', 'nervous', 'terrified', 'frightened', 'frightening',
                'concerned', 'concern', 'concerns', 'concerned', 'fear', 'fears', 'fearing', 'uneasy',
                'uneasiness', 'apprehensive', 'apprehension', 'panic', 'panicked', 'panicking', 'dread',
                'dreadful', 'horror', 'horrified', 'horrifying', 'alarming', 'alarmed', 'alarm',
                'stress', 'stressed', 'stressful', 'pressure', 'pressured', 'uncertain', 'uncertainty'
            ],
            'surprise': [
                'surprised', 'shocked', 'amazed', 'astonished', 'wow', 'incredible', 'unbelievable',
                'unexpected', 'unexpectedly', 'sudden', 'suddenly', 'surprise', 'surprising', 'surprisingly',
                'astounding', 'astounded', 'stunned', 'stunning', 'stun', 'baffled', 'baffling', 'bewildered',
                'bewildering', 'startled', 'startling', 'jaw-dropping', 'jaw dropping', 'remarkable'
            ],
            'disgust': [
                'disgusted', 'revolted', 'sick', 'gross', 'nasty', 'disgusting', 'revolting', 'repulsive',
                'repulsed', 'appalled', 'appalling', 'horrified', 'horrifying', 'sickening', 'sickened',
                'offensive', 'offended', 'offending', 'distasteful', 'unpleasant', 'unpleasantness',
                'filthy', 'dirty', 'unclean', 'unsanitary', 'contaminated', 'contamination', 'vile'
            ]
        }
        
        # Normalize text
        text_lower = str(text).lower()
        
        # Remove punctuation for better matching
        import re
        text_clean = re.sub(r'[^\w\s]', ' ', text_lower)
        words = text_clean.split()
        
        # Use word boundaries for better matching
        emotion_scores = {}
        
        for emotion, keywords in emotions.items():
            score = 0
            for keyword in keywords:
                # Check for exact word matches (word boundaries) - this is more accurate
                pattern = r'\b' + re.escape(keyword) + r'\b'
                matches = len(re.findall(pattern, text_lower))
                if matches > 0:
                    score += matches
                    # Add extra weight for strong emotional words
                    if keyword in ['worst', 'terrible', 'horrible', 'awful', 'disgusting', 'furious', 
                                   'ecstatic', 'thrilled', 'devastated', 'terrified', 'amazing', 'excellent']:
                        score += 0.5
            
            emotion_scores[emotion] = score
        
        # Calculate percentages (0-1 range for frontend compatibility)
        total = sum(emotion_scores.values())
        if total > 0:
            emotion_scores = {k: round(v / total, 4) for k, v in emotion_scores.items()}
        else:
            # If no matches found, default to neutral
            emotion_scores = {k: 0.0 for k in emotion_scores.keys()}
        
        # Find dominant emotion
        dominant_emotion = max(emotion_scores.items(), key=lambda x: x[1])[0] if total > 0 else None
        
        # Return in 0-1 format (frontend will multiply by 100 for display)
        return {
            'emotions': emotion_scores,
            'dominant_emotion': dominant_emotion
        }
    
    def extract_bigrams_trigrams(self, texts, n=20):
        """Extract important bigrams and trigrams"""
        from sklearn.feature_extraction.text import CountVectorizer
        
        # Bigrams
        bigram_vectorizer = CountVectorizer(ngram_range=(2, 2), max_features=n, min_df=2)
        try:
            bigram_counts = bigram_vectorizer.fit_transform(texts)
            bigram_features = bigram_vectorizer.get_feature_names_out()
            bigram_scores = np.array(bigram_counts.sum(axis=0)).flatten()
            
            bigrams = list(zip(bigram_features, bigram_scores))
            bigrams.sort(key=lambda x: x[1], reverse=True)
            
            # Trigrams
            trigram_vectorizer = CountVectorizer(ngram_range=(3, 3), max_features=n, min_df=2)
            trigram_counts = trigram_vectorizer.fit_transform(texts)
            trigram_features = trigram_vectorizer.get_feature_names_out()
            trigram_scores = np.array(trigram_counts.sum(axis=0)).flatten()
            
            trigrams = list(zip(trigram_features, trigram_scores))
            trigrams.sort(key=lambda x: x[1], reverse=True)
            
            return {
                'bigrams': [{'phrase': phrase, 'count': int(count)} for phrase, count in bigrams[:n]],
                'trigrams': [{'phrase': phrase, 'count': int(count)} for phrase, count in trigrams[:n]]
            }
        except:
            return {'bigrams': [], 'trigrams': []}
    
    def extract_phrases_patterns(self, texts):
        """Extract common phrases and patterns"""
        patterns = {
            'complaints': r'(not|never|no|can\'t|cannot|won\'t|wouldn\'t)',
            'praise': r'(excellent|great|amazing|wonderful|perfect|love|best)',
            'questions': r'\?',
            'exclamations': r'!',
            'negations': r'(not|no|never|none|nothing|nobody)',
        }
        
        results = {}
        all_text = ' '.join(texts).lower()
        
        for pattern_name, pattern in patterns.items():
            matches = re.findall(pattern, all_text, re.IGNORECASE)
            results[pattern_name] = len(matches)
        
        return results

