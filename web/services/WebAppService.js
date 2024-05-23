const axios = require('axios');

class WebAppService {

    // The constructor
    constructor() {
        // Set the base URL for the API
        this.baseURL = 'https://api.pi-nexus.com';
    }

    // The function to authenticate a user
    async authenticate(username, password) {
        // Validate the username and password
        if (!username || !password) {
            throw new Error('Invalid username or password');
        }

        // Authenticate the user
        const response = await axios.post(`${this.baseURL}/auth`, {
            username: username,
            password: password
        });

        // Return the user
        return response.data;
    }

    // The function to get the balance of a user
    async getBalance(user) {
        // Get the balance of the user
        const response = await axios.get(`${this.baseURL}/balance/${user.id}`, {
            headers: {
                'Authorization': `Bearer ${user.token}`
            }
        });

        // Return the balance
        return response.data;
    }

    // The function to transfer assets
    async transfer(user, recipient, amount) {
        // Validate the recipient and amount
        if (!recipient || !amount) {
            throw new Error('Invalid recipient or amount');
        }

        // Transfer the assets
        const response = await axios.post(`${this.baseURL}/transfer`, {
            senderId: user.id,
            recipient,
            amount
        }, {
            headers: {
                'Authorization': `Bearer ${user.token}`
            }
        });

        // Return the transaction
        return response.data;
    }

}

module.exports = WebAppService;
