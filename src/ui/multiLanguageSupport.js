// multiLanguageSupport.js

const fs = require('fs');
const path = require('path');

// Supported languages
const supportedLanguages = ['en', 'es', 'fr', 'de', 'zh'];

// Load language files
function loadLanguageFile(language) {
    const filePath = path.join(__dirname, 'languages', `${language}.json`);
    if (fs.existsSync(filePath)) {
        return JSON.parse(fs.readFileSync(filePath, 'utf-8'));
    } else {
        console.warn(`Language file for ${language} not found. Falling back to English.`);
        return loadLanguageFile('en'); // Fallback to English
    }
}

// Language data
let currentLanguage = 'en';
let translations = loadLanguageFile(currentLanguage);

// Change language
function changeLanguage(language) {
    if (supportedLanguages.includes(language)) {
        currentLanguage = language;
        translations = loadLanguageFile(currentLanguage);
        console.log(`Language changed to ${currentLanguage}`);
    } else {
        console.warn(`Language ${language} is not supported.`);
    }
}

// Translate a key
function translate(key) {
    return translations[key] || key; // Return the key if translation is not found
}

// Get the current language
function getCurrentLanguage() {
    return currentLanguage;
}

// Exporting functions for use in other modules
module.exports = {
    changeLanguage,
    translate,
    getCurrentLanguage,
};
