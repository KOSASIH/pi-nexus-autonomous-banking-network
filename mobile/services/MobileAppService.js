const MobileAppUtils = require('../utils/MobileAppUtils');

class MobileAppService {

    // The constructor
    constructor() {
        this.mobileAppUtils = new MobileAppUtils();
    }

    // The function to authenticate a user
    async authenticate(username, password) {
        // Validate the username and password
        if (!username || !password) {
            throw new Error('Invalid username or password');
        }

        // Authenticate the user
        const user = await this.mobileAppUtils.authenticate(username, password);

        // Return the user
        return user;
    }

    // The function to get the balance of a user
    async getBalance(user) {
        // Get the balance of the user
        const balance = await this.mobileAppUtils.getBalance(user);

        // Return the balance
        return balance;
    }

    // The function to transfer assets
    async transfer(user, recipient, amount) {
        // Validate the recipient and amount
        if (!recipient || !amount) {
            throw new Error('Invalid recipient or amount');
        }

        // Transfer the assets
        const transaction = await this.mobileAppUtils.transfer(user, recipient, amount);

        // Return the transaction
        return transaction;
    }

    // The function to subscribe to push notifications
    async subscribeToPushNotifications(user) {
        // Subscribe the user to push notifications
        const subscription = await this.mobileAppUtils.subscribeToPushNotifications(user);

        // Return the subscription
        return subscription;
    }

    // The function to unsubscribe from push notifications
    async unsubscribeFromPushNotifications(user) {
        // Unsubscribe the user from push notifications
        const success = await this.mobileAppUtils.unsubscribeFromPushNotifications(user);

        // Return the success status
        return success;
    }

}

module.exports = MobileAppService;
