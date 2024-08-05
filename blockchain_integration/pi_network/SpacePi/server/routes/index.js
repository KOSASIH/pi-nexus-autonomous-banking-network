const express = require('express');
const router = express.Router();
const launchesRouter = require('./space-x/launches');
const merchandiseRouter = require('./space-x/merchandise');
const usersRouter = require('./users');

router.use('/space-x/launches', launchesRouter);
router.use('/space-x/merchandise', merchandiseRouter);
router.use('/users', usersRouter);

module.exports = router;
