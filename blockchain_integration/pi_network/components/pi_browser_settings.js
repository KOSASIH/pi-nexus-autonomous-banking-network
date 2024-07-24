import React, { useState } from 'eact';
import { PiBrowser } from '@pi-network/pi-browser-sdk';

const PiBrowserSettings = () => {
  const [theme, setTheme] = useState('light');
  const [language, setLanguage] = useState('en');
  const [apiKey, setApiKey] = useState('');

  const handleThemeChange = e => {
    setTheme(e.target.value);
    PiBrowser.setTheme(e.target.value);
  };

  const handleLanguageChange = e => {
    setLanguage(e.target.value);
    PiBrowser.setLanguage(e.target.value);
  };

  const handleApiKeyChange = e => {
    setApiKey(e.target.value);
    PiBrowser.setApiKey(e.target.value);
  };

  return (
    <div>
      <h1>Pi Browser Settings</h1>
      <section>
        <h2>Theme</h2>
        <select value={theme} onChange={handleThemeChange}>
          <option value="light">Light Mode</option>
          <option value="dark">Dark Mode</option>
        </select>
      </section>
      <section>
        <h2>Language</h2>
        <select value={language} onChange={handleLanguageChange}>
          <option value="en">English</option>
          <option value="es">Spanish</option>
          <option value="fr">French</option>
        </select>
      </section>
      <section>
        <h2>API Key</h2>
        <input
          type="text"
          value={apiKey}
          onChange={handleApiKeyChange}
          placeholder="Enter API key"
        />
      </section>
    </div>
  );
};

export default PiBrowserSettings;
