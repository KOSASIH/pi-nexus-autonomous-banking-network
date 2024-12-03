// tests/customization.test.js
const SettingsManager = require('../customization/settings');
const ThemeManager = require('../customization/themes');

describe('Customization Module', () => {
    let settingsManager;
    let themeManager;

    beforeEach(() => {
        settingsManager = new SettingsManager();
        themeManager = new ThemeManager();
    });

    test('should create user settings correctly', () => {
        const userSettings = settingsManager.createUserSettings('user123');
        expect(userSettings.getSettings()).toHaveProperty('language', 'en');
    });

    test('should add a new theme correctly', () => {
        const theme = themeManager.addTheme('Dark Mode', { backgroundColor: '#000', textColor: '#fff' });
        expect(theme.name).toBe('Dark Mode');
    });
});
