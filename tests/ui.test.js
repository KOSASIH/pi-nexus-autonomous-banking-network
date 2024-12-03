// tests/ui.test.js
const UI = require('../ui'); // Assuming you have a UI module

describe('UI Module', () => {
    test('should render the main page correctly', () => {
        const mainPage = UI.renderMainPage();
        expect(mainPage).toMatchSnapshot();
    });

    test('should show error message on invalid input', () => {
        const errorMessage = UI.validateInput('');
        expect(errorMessage).toBe('Input cannot be empty');
    });
});
