import React, { useState, useEffect } from 'eact';
import { BrowserRouter, Route, Switch } from 'eact-router-dom';
import PiBrowserNeuralNetworks from './PiBrowserNeuralNetworks';
import PiBrowserBlockchainExplorer from './PiBrowserBlockchainExplorer';
import PiBrowserArtificialIntelligence from './PiBrowserArtificialIntelligence';
import PiBrowserVirtualReality from './PiBrowserVirtualReality';
import PiBrowserCybersecurity from './PiBrowserCybersecurity';
import PiBrowserNewsFeed from './PiBrowserNewsFeed';
import PiBrowserUtilities from './PiBrowserUtilities';
import PiBrowserBlockchain from './PiBrowserBlockchain';
import PiBrowserNeuralNetworkExplorer from './PiBrowserNeuralNetworkExplorer';
import PiBrowserAIAssistant from './PiBrowserAIAssistant';

const App = () => {
  const [theme, setTheme] = useState('light');
  const [language, setLanguage] = useState('en');

  useEffect(() => {
    // Initialize theme and language settings
    const storedTheme = localStorage.getItem('theme');
    const storedLanguage = localStorage.getItem('language');
    if (storedTheme) setTheme(storedTheme);
    if (storedLanguage) setLanguage(storedLanguage);
  }, []);

  const handleThemeChange = (theme) => {
    setTheme(theme);
    localStorage.setItem('theme', theme);
  };

  const handleLanguageChange = (language) => {
    setLanguage(language);
    localStorage.setItem('language', language);
  };

  return (
    <BrowserRouter>
      <Switch>
        <Route path="/" exact component={PiBrowserNewsFeed} />
        <Route path="/neural-networks" component={PiBrowserNeuralNetworks} />
        <Route path="/blockchain-explorer" component={PiBrowserBlockchainExplorer} />
        <Route path="/artificial-intelligence" component={PiBrowserArtificialIntelligence} />
        <Route path="/virtual-reality" component={PiBrowserVirtualReality} />
        <Route path="/cybersecurity" component={PiBrowserCybersecurity} />
        <Route path="/utilities" component={PiBrowserUtilities} />
        <Route path="/blockchain" component={PiBrowserBlockchain} />
        <Route path="/neural-network-explorer" component={PiBrowserNeuralNetworkExplorer} />
        <Route path="/ai-assistant" component={PiBrowserAIAssistant} />
      </Switch>
      <footer>
        <p>Copyright 2024 Pi Browser</p>
        <button onClick={() => handleThemeChange('dark')}>Dark Mode</button>
        <button onClick={() => handleThemeChange('light')}>Light Mode</button>
        <select value={language} onChange={(e) => handleLanguageChange(e.target.value)}>
          <option value="en">English</option>
          <option value="es">Spanish</option>
          <option value="fr">French</option>
        </select>
      </footer>
    </BrowserRouter>
  );
};

export default App;
