import React from 'react';
import ReactDOM from 'react-dom';
import { AIProvider } from './contexts/AIContext';
import { UserProvider } from './contexts/UserContext';
import AppRouter from './AppRouter';

const App = () => {
    return (
        <AIProvider>
            <UserProvider>
                <AppRouter />
            </UserProvider>
        </AIProvider>
    );
};

ReactDOM.render(<App />, document.getElementById('root'));
