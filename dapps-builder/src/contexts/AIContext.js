import { createContext, useState, useEffect } from 'react';
import useAI from '../hooks/useAI';

const AIContext = createContext();

const AIProvider = ({ children }) => {
    const { model, loading, error, trainModel, loadModel, predict, saveModel } = useAI();

    return (
        <AIContext.Provider value={{ model, loading, error, trainModel, loadModel, predict, saveModel }}>
            {children}
        </AIContext.Provider>
    );
};

export { AIProvider, AIContext };
