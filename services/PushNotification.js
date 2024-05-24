// services/PushNotification.js
const Expo = require("expo-server-sdk");

const pushNotification = async (user, message) => {
  const expo = new Expo();
  const tokens = await getPushTokensForUser(user);
  const notifications = tokens.map((token) => ({
    to: token,
    sound: "default",
    title: "Transaction Alert",
    body: message,
  }));
  await expo.sendPushNotificationsAsync(notifications);
};

const getPushTokensForUser = async (user) => {
  // Implement logic to retrieve push tokens for the user
  // For example, using a database or a token storage service
  return ["ExponentPushToken[xxxxxxxxxxxxxxxxxxxxxx]"];
};

module.exports = { sendNotification: pushNotification };
