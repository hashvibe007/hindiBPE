import React, { useState, useEffect } from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './styles.css';

const TrainingProgress = () => {
    const [progress, setProgress] = useState(null);
    const [selectedMetric, setSelectedMetric] = useState('compression_ratios');
    const [ws, setWs] = useState(null);

    useEffect(() => {
        // Initial fetch
        fetchTrainingProgress();

        // Set up WebSocket connection
        const websocket = new WebSocket('ws://localhost:8000/ws');
        
        websocket.onmessage = (event) => {
            const update = JSON.parse(event.data);
            if (update.type === 'training_update') {
                setProgress(prevProgress => ({
                    ...prevProgress,
                    ...update.data
                }));
            }
        };

        setWs(websocket);

        return () => {
            if (websocket) {
                websocket.close();
            }
        };
    }, []);

    // Add auto-refresh for polling updates
    useEffect(() => {
        const interval = setInterval(() => {
            fetchTrainingProgress();
        }, 5000);  // Poll every 5 seconds

        return () => clearInterval(interval);
    }, []);

    const fetchTrainingProgress = async () => {
        try {
            const response = await fetch('http://localhost:8000/training-progress');
            const data = await response.json();
            setProgress(data);
        } catch (error) {
            console.error('Error fetching training progress:', error);
        }
    };

    const formatData = (metrics) => {
        if (!metrics) return [];
        const steps = Array.from({ length: metrics.vocab_sizes.length }, (_, i) => i);
        return steps.map(i => ({
            step: i,
            'Vocabulary Size': metrics.vocab_sizes[i],
            'Compression Ratio': metrics.compression_ratios[i],
            'Merge Frequency': metrics.merge_frequencies[i],
            'Unique Tokens': metrics.unique_tokens[i]
        }));
    };

    const metricConfigs = {
        vocab_sizes: {
            name: 'Vocabulary Size',
            color: '#8884d8'
        },
        compression_ratios: {
            name: 'Compression Ratio',
            color: '#82ca9d'
        },
        merge_frequencies: {
            name: 'Merge Frequency',
            color: '#ffc658'
        },
        unique_tokens: {
            name: 'Unique Tokens',
            color: '#ff7300'
        }
    };

    const BaseVocabStats = () => (
        <div className="base-vocab-stats">
            <h3>Base Vocabulary (वर्णमाला)</h3>
            {progress?.base_vocab_stats && (
                <div className="base-vocab-grid">
                    <div className="vocab-stat">
                        <span className="vocab-label">व्यंजन (Consonants):</span>
                        <span className="vocab-value">{progress.base_vocab_stats.vyanjan}</span>
                    </div>
                    <div className="vocab-stat">
                        <span className="vocab-label">स्वर (Vowels):</span>
                        <span className="vocab-value">{progress.base_vocab_stats.swar}</span>
                    </div>
                    <div className="vocab-stat">
                        <span className="vocab-label">मात्राएँ (Matras):</span>
                        <span className="vocab-value">{progress.base_vocab_stats.matras}</span>
                    </div>
                    <div className="vocab-stat">
                        <span className="vocab-label">विशेष (Special):</span>
                        <span className="vocab-value">{progress.base_vocab_stats.special}</span>
                    </div>
                    <div className="vocab-stat total">
                        <span className="vocab-label">कुल (Total):</span>
                        <span className="vocab-value">{progress.base_vocab_stats.total}</span>
                    </div>
                </div>
            )}
        </div>
    );

    const TrainingStatus = ({ progress }) => {
        if (!progress) return null;

        const isTraining = progress.steps?.length < progress.target_vocab_size;
        
        return (
            <div className={`training-status ${isTraining ? 'active' : ''}`}>
                <div className="status-indicator"></div>
                <span>
                    {isTraining ? 'Training in Progress' : 'Training Complete'}
                </span>
                <div className="progress-details">
                    <span>{progress.steps?.length || 0} / {progress.target_vocab_size}</span>
                </div>
            </div>
        );
    };

    return (
        <div className="training-progress">
            <h2>Training Progress</h2>
            
            <BaseVocabStats />
            
            <div className="metric-selector">
                <select 
                    value={selectedMetric} 
                    onChange={(e) => setSelectedMetric(e.target.value)}
                >
                    {Object.entries(metricConfigs).map(([key, config]) => (
                        <option key={key} value={key}>{config.name}</option>
                    ))}
                </select>
            </div>

            {progress && progress.metrics && (
                <div className="vocab-summary">
                    <div className="vocab-stat">
                        <span>Base Vocabulary:</span>
                        <span>{progress.base_vocab_stats.total}</span>
                    </div>
                    <div className="vocab-stat">
                        <span>Learned Tokens:</span>
                        <span>{progress.metrics.learned_vocab_sizes[progress.metrics.learned_vocab_sizes.length - 1]}</span>
                    </div>
                    <div className="vocab-stat">
                        <span>Total Vocabulary:</span>
                        <span>{progress.metrics.vocab_sizes[progress.metrics.vocab_sizes.length - 1]}</span>
                    </div>
                </div>
            )}

            {progress && progress.metrics && (
                <div className="chart-container">
                    <ResponsiveContainer width="100%" height={300}>
                        <LineChart data={formatData(progress.metrics)}>
                            <CartesianGrid strokeDasharray="3 3" />
                            <XAxis dataKey="step" />
                            <YAxis />
                            <Tooltip />
                            <Legend />
                            <Line
                                type="monotone"
                                dataKey={metricConfigs[selectedMetric].name}
                                stroke={metricConfigs[selectedMetric].color}
                                dot={false}
                            />
                        </LineChart>
                    </ResponsiveContainer>
                </div>
            )}

            <div className="training-summary">
                <h3>Training Summary</h3>
                {progress && (
                    <div className="summary-stats">
                        <div>Initial Vocabulary: {progress.initial_vocab_size}</div>
                        <div>Target Vocabulary: {progress.target_vocab_size}</div>
                        <div>Current Step: {progress.steps?.length || 0}</div>
                    </div>
                )}
            </div>
        </div>
    );
};

export default TrainingProgress; 