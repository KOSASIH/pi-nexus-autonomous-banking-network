// api/endpoints.js
const express = require('express');
const APIIntegration = require('./integration');

const router = express.Router();
const apiClient = new APIIntegration('https://api.example.com'); // Replace with actual API base URL

// Middleware for logging requests
router.use((req, res, next) => {
    console.log(`${req.method} request for '${req.url}'`);
    next();
});

// Example endpoint to get user data
router.get('/users/:id', async (req, res) => {
    try {
        const userId = req.params.id;
        const userData = await apiClient.get(`/users/${userId}`);
        res.status(200).json(userData);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Example endpoint to create a new user
router.post('/users', async (req, res) => {
    try {
        const newUser = req.body;
        const createdUser = await apiClient.post('/users', newUser);
        res.status(201).json(createdUser);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Example endpoint to update user data
router.put('/users/:id', async (req, res) => {
    try {
        const userId = req.params.id;
        const updatedUser = req.body;
        const response = await apiClient.post(`/users/${userId}`, updatedUser);
        res.status(200).json(response);
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

// Example endpoint to delete a user
router.delete('/users/:id', async (req, res) => {
    try {
        const userId = req.params.id;
        await apiClient.delete(`/users/${userId}`);
        res.status(204).send();
    } catch (error) {
        res.status(500).json({ error: error.message });
    }
});

module.exports = router;
