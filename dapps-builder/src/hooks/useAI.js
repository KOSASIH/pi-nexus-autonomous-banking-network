import { useState, useEffect } from 'react';
import AIService from '../services/AIService';

const useAI = () => {
    const [model, setModel] = useState(null);
    const [loading, setLoading] = useState(false);
    const [error, setError] = useState(null);

    const trainModel = async (data) => {
        setLoading(true);
        setError(null);
        try {
            const trainedModel = await AIService.train(data);
            setModel(trainedModel);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const loadModel = async (modelPath) => {
        setLoading(true);
        setError(null);
        try {
            const loadedModel = await AIService.loadModel(modelPath);
            setModel(loadedModel);
        } catch (err) {
            setError(err.message);
        } finally {
            setLoading(false);
        }
    };

    const predict = async (input) => {
        if (!model) {
            throw new Error('Model not loaded or trained.');
        }
        return await AIService.predict(input);
    };

    const saveModel = async (path) => {
        if (!model) {
            throw new Error('Model not trained or loaded.');
        }
        await AIService.saveModel(path);
    };

    return {
        model,
        loading,
        error,
        trainModel,
        loadModel,
        predict,
        saveModel,
    };
};

export default useAI;
