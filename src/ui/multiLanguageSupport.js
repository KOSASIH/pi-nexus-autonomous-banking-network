// multiLanguageSupport.js

const fs = require('fs');
const path = require('path');

// Supported languages
const supportedLanguages = {
    'en': 'English',
    'es': 'Spanish',
    'fr': 'French',
    'de': 'German',
    'zh': 'Chinese',
    'it': 'Italian',
    'pt': 'Portuguese',
    'ru': 'Russian',
    'ja': 'Japanese',
    'ko': 'Korean',
    'nl': 'Dutch',
    'sv': 'Swedish',
    'da': 'Danish',
    'fi': 'Finnish',
    'tr': 'Turkish',
    'ar': 'Arabic',
    'hi': 'Hindi',
    'th': 'Thai',
    'pl': 'Polish',
    'cs': 'Czech',
    'el': 'Greek',
    'id': 'Indonesian',
    'vi': 'Vietnamese'
    'su': 'Sundanese'
};

// Cache for translations
const translationCache = {};

// Load language files
function loadLanguageFile(language) {
    if (translationCache[language]) {
        return translationCache[language];
    }

    const filePath = path.join(__dirname, 'languages', `${language}.json`);
    if (fs.existsSync(filePath)) {
        const translations = JSON.parse(fs.readFileSync(filePath, 'utf-8'));
        translationCache[language] = translations;
        return translations;
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
    if (supportedLanguages[language]) {
        currentLanguage = language;
        translations = loadLanguageFile(currentLanguage);
        console.log(`Language changed to ${supportedLanguages[language]}`);
    } else {
        console.warn(`Language ${language} is not supported.`);
    }
}

// Translate a key with fallback
function translate(key, count) {
    let translation = translations[key];

    if (typeof translation === 'undefined') {
        console.warn(`Translation for key "${key}" not found in ${currentLanguage}. Falling back to English.`);
        translation = loadLanguageFile('en')[key] || key; // Fallback to English or return the key itself
    }

    // Handle pluralization if count is provided
    if (count !== undefined) {
        if (typeof translation === 'object' && translation.plural) {
            return count ===  1 ? translation.singular : translation.plural;
        }
    }

    return translation; // Return the translation
}

// Get the current language
function getCurrentLanguage() {
    return currentLanguage;
}

// List all available languages
function listAvailableLanguages() {
    return Object.entries(supportedLanguages).map(([code, name]) => `${code}: ${name}`).join(', ');
}

// Reset to default language
function resetLanguage() {
    currentLanguage = 'en';
    translations = loadLanguageFile(currentLanguage);
    console.log(`Language reset to default: ${supportedLanguages[currentLanguage]}`);
}

// Get all translations for the current language
function getAllTranslations() {
    return translations;
}

// Dynamically load a new language file
function loadNewLanguage(language) {
    if (!supportedLanguages[language]) {
        console.warn(`Language ${language} is not supported.`);
        return;
    }
    loadLanguageFile(language);
    console.log(`Language file for ${supportedLanguages[language]} loaded successfully.`);
}

// Export functions
module.exports = {
    changeLanguage,
    translate,
    getCurrentLanguage,
    listAvailableLanguages,
    resetLanguage,
    getAllTranslations,
    loadNewLanguage
};
