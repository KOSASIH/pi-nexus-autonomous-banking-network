const biometric = require('biometric-authentication');

async function authenticateUser(userInput) {
  const result = await biometric.authenticate(userInput);
  return result;
}

module.exports = { authenticateUser };
