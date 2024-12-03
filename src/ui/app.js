// src/ui/app.js

document.addEventListener('DOMContentLoaded', () => {
    const registerForm = document.getElementById('register-form');
    const loginForm = document.getElementById('login-form');

    // Function to display messages
    function displayMessage(element, message, isSuccess) {
        const messageElement = document.createElement('div');
        messageElement.textContent = message;
        messageElement.className = isSuccess ? 'success-message' : 'error-message';
        element.appendChild(messageElement);
        setTimeout(() => {
            messageElement.remove();
        }, 3000);
    }

    // Validate form inputs
    function validateForm(username, password) {
        if (!username || !password) {
            return 'Username and password are required.';
        }
        if (password.length < 6) {
            return 'Password must be at least 6 characters long.';
        }
        return null;
    }

    // Handle user registration
    if (registerForm) {
        registerForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('register-username').value;
            const password = document.getElementById('register-password').value;

            const validationError = validateForm(username, password);
            if (validationError) {
                displayMessage(registerForm, validationError, false);
                return;
            }

            try {
                const response = await fetch('/register', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ username, password }),
                });

                const data = await response.json();
                if (response.ok) {
                    displayMessage(registerForm, 'Registrationsuccessful!', true);
                    registerForm.reset();
                } else {
                    displayMessage(registerForm, `Error: ${data.error}`, false);
                }
            } catch (error) {
                console.error('Error:', error);
                displayMessage(registerForm, 'An error occurred during registration.', false);
            }
        });
    }

    // Handle user login
    if (loginForm) {
        loginForm.addEventListener('submit', async (e) => {
            e.preventDefault();
            const username = document.getElementById('login-username').value;
            const password = document.getElementById('login-password').value;

            const validationError = validateForm(username, password);
            if (validationError) {
                displayMessage(loginForm, validationError, false);
                return;
            }

            try {
                const response = await fetch('/login', {
                    method: 'POST',
                    headers: {
                        'Content-Type': 'application/json',
                    },
                    body: JSON.stringify({ username, password }),
                });

                const data = await response.json();
                if (response.ok) {
                    displayMessage(loginForm, 'Login successful! Token: ' + data.token, true);
                    loginForm.reset();
                } else {
                    displayMessage(loginForm, `Error: ${data.error}`, false);
                }
            } catch (error) {
                console.error('Error:', error);
                displayMessage(loginForm, 'An error occurred during login.', false);
            }
        });
    }
});
