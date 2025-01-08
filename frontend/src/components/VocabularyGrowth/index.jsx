import React from 'react';
import { LineChart, Line, XAxis, YAxis, CartesianGrid, Tooltip, Legend, ResponsiveContainer } from 'recharts';
import './styles.css';

const VocabularyGrowth = ({ vocabGrowth }) => {
    const formatData = (data) => {
        return data.merge_steps.map((step, idx) => ({
            step,
            frequency: data.frequencies[idx],
            token: data.tokens[idx],
            composition: data.compositions[idx].join(' + ')
        }));
    };

    return (
        <div className="vocab-growth">
            <h3>Vocabulary Growth Analysis</h3>
            
            <div className="chart-container">
                <h4>Token Frequencies Over Time</h4>
                <ResponsiveContainer width="100%" height={300}>
                    <LineChart data={formatData(vocabGrowth)}>
                        <CartesianGrid strokeDasharray="3 3" />
                        <XAxis dataKey="step" label="Merge Steps" />
                        <YAxis label="Frequency" />
                        <Tooltip 
                            content={({ payload, label }) => {
                                if (payload && payload.length) {
                                    const data = payload[0].payload;
                                    return (
                                        <div className="custom-tooltip">
                                            <p>Step: {data.step}</p>
                                            <p>Token: {data.token}</p>
                                            <p>Frequency: {data.frequency}</p>
                                            <p>Composition: {data.composition}</p>
                                        </div>
                                    );
                                }
                                return null;
                            }}
                        />
                        <Line type="monotone" dataKey="frequency" stroke="#8884d8" />
                    </LineChart>
                </ResponsiveContainer>
            </div>

            <div className="token-list">
                <h4>Recent Token Additions</h4>
                <div className="token-grid">
                    {vocabGrowth.tokens.slice(-10).map((token, idx) => (
                        <div key={idx} className="token-card">
                            <div className="token-text">{token}</div>
                            <div className="token-details">
                                <span>Freq: {vocabGrowth.frequencies[idx]}</span>
                                <span>Step: {vocabGrowth.merge_steps[idx]}</span>
                                <div className="token-composition">
                                    {vocabGrowth.compositions[idx].join(' + ')}
                                </div>
                            </div>
                        </div>
                    ))}
                </div>
            </div>
        </div>
    );
};

export default VocabularyGrowth; 