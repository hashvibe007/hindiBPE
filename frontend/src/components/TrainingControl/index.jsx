import React, { useState } from 'react';
import './styles.css';

const TrainingControl = ({ onTrainingStart }) => {
    const [maxSentences, setMaxSentences] = useState(10000);
    const [vocabSize, setVocabSize] = useState(10000);
    const [isStarting, setIsStarting] = useState(false);

    const handleStartTraining = async () => {
        try {
            setIsStarting(true);
            const response = await fetch('http://localhost:8000/start-training', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({
                    max_sentences: maxSentences,
                    vocab_size: vocabSize
                }),
            });

            if (!response.ok) {
                throw new Error('Failed to start training');
            }

            if (onTrainingStart) {
                onTrainingStart();
            }
        } catch (error) {
            console.error('Error starting training:', error);
        } finally {
            setIsStarting(false);
        }
    };

    return (
        <div className="training-control">
            <h3>Training Control</h3>
            <div className="control-grid">
                <div className="control-item">
                    <label>Max Sentences:</label>
                    <input
                        type="number"
                        value={maxSentences}
                        onChange={(e) => setMaxSentences(parseInt(e.target.value))}
                        min="100"
                        max="100000"
                    />
                </div>
                <div className="control-item">
                    <label>Vocabulary Size:</label>
                    <input
                        type="number"
                        value={vocabSize}
                        onChange={(e) => setVocabSize(parseInt(e.target.value))}
                        min="1000"
                        max="50000"
                    />
                </div>
                <button
                    className="start-button"
                    onClick={handleStartTraining}
                    disabled={isStarting}
                >
                    {isStarting ? 'Starting Training...' : 'Start Training'}
                </button>
            </div>
        </div>
    );
};

export default TrainingControl; 