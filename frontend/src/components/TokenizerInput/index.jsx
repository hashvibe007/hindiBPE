import React, { useState, useEffect } from 'react';
import './styles.css';
import TrainingProgress from '../TrainingProgress';
import VocabularyGrowth from '../VocabularyGrowth';

const TokenizerInput = () => {
    const [inputText, setInputText] = useState('');
    const [tokenData, setTokenData] = useState(null);
    const [isLoading, setIsLoading] = useState(false);
    const [error, setError] = useState(null);
    const [selectedToken, setSelectedToken] = useState(null);
    const [trainingStats, setTrainingStats] = useState(null);

    const getTokenColor = (type) => {
        const colors = {
            consonant: '#e9d5ff',  // Purple
            vowel: '#bfdbfe',      // Blue
            matra: '#fecaca',      // Red
            special: '#fef08a',    // Yellow
            compound: '#bbf7d0'    // Green
        };
        return colors[type] || '#e5e7eb';  // Default gray
    };

    const handleTokenize = async () => {
        try {
            setIsLoading(true);
            setError(null);
            setSelectedToken(null);
            
            const response = await fetch('http://localhost:8000/tokenize', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ text: inputText }),
            });

            if (!response.ok) {
                const errorData = await response.json();
                throw new Error(errorData.message || 'Tokenization failed');
            }

            const data = await response.json();
            console.log('Tokenization Data:', data);  // Debug log
            console.log('Original Tokens:', data.original_tokens);
            console.log('Encoded Tokens:', data.original_encoded_tokens);
            console.log('BPE Tokens:', data.bpe_tokens);
            console.log('Token Numbers:', data.token_numbers);
            setTokenData(data);
        } catch (error) {
            setError(error.message);
        } finally {
            setIsLoading(false);
        }
    };

    const fetchTrainingStats = async () => {
        try {
            const response = await fetch('http://localhost:8000/training-stats');
            const data = await response.json();
            setTrainingStats(data);
        } catch (error) {
            console.error('Error fetching training stats:', error);
        }
    };

    useEffect(() => {
        fetchTrainingStats();
    }, []);

    const TokenTypeLegend = () => (
        <div className="token-type-legend">
            <div className="legend-item">
                <span className="legend-color" style={{backgroundColor: getTokenColor('consonant')}}></span>
                <span>Consonant</span>
            </div>
            <div className="legend-item">
                <span className="legend-color" style={{backgroundColor: getTokenColor('vowel')}}></span>
                <span>Vowel</span>
            </div>
            <div className="legend-item">
                <span className="legend-color" style={{backgroundColor: getTokenColor('matra')}}></span>
                <span>Matra</span>
            </div>
            <div className="legend-item">
                <span className="legend-color" style={{backgroundColor: getTokenColor('special')}}></span>
                <span>Special</span>
            </div>
            <div className="legend-item">
                <span className="legend-color" style={{backgroundColor: getTokenColor('compound')}}></span>
                <span>Compound</span>
            </div>
        </div>
    );

    return (
        <div className="app-container">
            <div className="training-section">
                <TrainingProgress />
                {trainingStats && trainingStats.vocab_growth && (
                    <VocabularyGrowth vocabGrowth={trainingStats.vocab_growth} />
                )}
            </div>
            <div className="tokenizer-container">
                <h1>Hindi BPE Tokenizer</h1>
                
                <div className="input-section">
                    <textarea
                        placeholder="यहाँ हिंदी टेक्स्ट दर्ज करें..."
                        value={inputText}
                        onChange={(e) => setInputText(e.target.value)}
                        className="hindi-input"
                    />
                    <button 
                        onClick={handleTokenize} 
                        className="tokenize-button"
                        disabled={isLoading || !inputText.trim()}
                    >
                        {isLoading ? 'Tokenizing...' : 'Tokenize'}
                    </button>
                </div>

                {error && (
                    <div className="error-message">
                        {error}
                    </div>
                )}

                {tokenData && tokenData.stats && (
                    <div className="stats-section">
                        <h2>Token Statistics</h2>
                        <div className="stats-grid">
                            <div className="stat-item">
                                <span className="stat-label">Original Characters:</span>
                                <span className="stat-value">{tokenData.stats.original_chars}</span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Token Count:</span>
                                <span className="stat-value">{tokenData.stats.token_count}</span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Unique Tokens:</span>
                                <span className="stat-value">{tokenData.stats.unique_tokens}</span>
                            </div>
                            <div className="stat-item">
                                <span className="stat-label">Compression Ratio:</span>
                                <span className="stat-value">{tokenData.stats.compression_ratio}:1</span>
                            </div>
                        </div>
                    </div>
                )}

                {tokenData && tokenData.token_details && (
                    <div className="tokens-section">
                        <h2>Tokens Analysis</h2>
                        <TokenTypeLegend />
                        <div className="tokens-flow">
                            {tokenData.token_details.map((token, index) => (
                                <div
                                    key={index}
                                    className={`token-chip ${selectedToken === index ? 'selected' : ''}`}
                                    style={{backgroundColor: getTokenColor(token.type)}}
                                    onClick={() => setSelectedToken(selectedToken === index ? null : index)}
                                >
                                    <span className="token-text">{token.token}</span>
                                    <span className="token-number">{tokenData.token_numbers[index]}</span>
                                    {selectedToken === index && tokenData.token_numbers && tokenData.token_numbers.length > index && (
                                        <div className="token-popup">
                                            <div>Length: {token.length}</div>
                                            <div>Type: {token.type}</div>
                                            <div>Unicode: {Array.from(token.token).map(char => 
                                                char.charCodeAt(0).toString(16)
                                            ).join(' ')}</div>
                                        </div>
                                    )}
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {tokenData && tokenData.merge_history && tokenData.merge_history.length > 0 && (
                    <div className="merge-history-section">
                        <h2>Recent Merge Operations</h2>
                        <div className="merge-list">
                            {tokenData.merge_history.map((merge, index) => (
                                <div key={index} className="merge-item">
                                    <span className="merge-pair">{merge.pair.join(' + ')}</span>
                                    <span className="merge-arrow">→</span>
                                    <span className="merge-result">{merge.new_token}</span>
                                    <span className="merge-freq">({merge.frequency} times)</span>
                                </div>
                            ))}
                        </div>
                    </div>
                )}

                {tokenData && tokenData.original_tokens && tokenData.original_encoded_tokens && tokenData.bpe_tokens && tokenData.token_numbers &&
                    tokenData.original_tokens.length === tokenData.original_encoded_tokens.length &&
                    tokenData.original_tokens.length === tokenData.bpe_tokens.length &&
                    tokenData.original_tokens.length === tokenData.token_numbers.length && (
                    <div className="token-comparison">
                        <h2>Token Comparison</h2>
                        <table>
                            <thead>
                                <tr>
                                    <th>Original Token</th>
                                    <th>Encoded Token</th>
                                    <th>BPE Token</th>
                                </tr>
                            </thead>
                            <tbody>
                                {tokenData.original_tokens.map((token, index) => (
                                    <tr key={index}>
                                        <td>{token}</td>
                                        <td>{Array.from(tokenData.original_encoded_tokens[index] || []).join(' ')}</td>
                                        <td>{tokenData.bpe_tokens[index]}</td>
                                    </tr>
                                ))}
                            </tbody>
                        </table>
                    </div>
                )}
            </div>
        </div>
    );
};

export default TokenizerInput; 