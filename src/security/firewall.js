// src/security/firewall.js
const rateLimit = require('express-rate-limit');
const express = require('express');

const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 100, // Limit each IP to 100 requests per windowMs
    message: 'Too many requests, please try again later.',
});

const whitelist = ['127.0.0.1', '::1']; // Add your allowed IPs here

function ipWhitelist(req, res, next) {
    if (!whitelist.includes(req.ip)) {
        return res.status(403).send('Forbidden');
    }
    next();
}

function applyFirewall(app) {
    app.use(limiter);
    app.use(ipWhitelist);
}

module.exports = applyFirewall;
