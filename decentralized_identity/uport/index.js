const uport = require('uport-credentials');
const Identity = require('./models/Identity');

const identityManager = async (req, res) => {
    const user = req.body.user;
    const identity = await Identity.create(user);
    const credentials = await uport.createCredentials(identity);
    res.json(credentials);
};

module.exports = identityManager;
