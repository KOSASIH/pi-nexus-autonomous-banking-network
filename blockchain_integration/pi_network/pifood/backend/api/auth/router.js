const express = require('express');
const router = express.Router();
const bcrypt = require('bcryptjs');
const jwt = require('jsonwebtoken');
const passport = require('passport');
const { Strategy } = require('passport-local');
const { OAuth2Strategy } = require('passport-oauth2');
const { GoogleStrategy } = require('passport-google-oauth20');
const { FacebookStrategy } = require('passport-facebook');
const User = require('../models/User');
const authConfig = require('../config/auth');

const localStrategy = new Strategy({
  usernameField: 'email',
  passwordField: 'password',
}, async (email, password, done) => {
  try {
    const user = await User.findOne({ email });
    if (!user) {
      return done(null, false, { message: 'Invalid email or password' });
    }
    const isValid = await bcrypt.compare(password, user.password);
    if (!isValid) {
      return done(null, false, { message: 'Invalid email or password' });
    }
    return done(null, user);
  } catch (err) {
    return done(err);
  }
});

const googleStrategy = new GoogleStrategy({
  clientID: authConfig.google.clientId,
  clientSecret: authConfig.google.clientSecret,
  callbackURL: authConfig.google.callbackURL,
}, async (accessToken, refreshToken, profile, done) => {
  try {
    const user = await User.findOne({ googleId: profile.id });
    if (user) {
      return done(null, user);
    }
    const newUser = new User({
      googleId: profile.id,
      email: profile.emails[0].value,
      name: profile.displayName,
    });
    await newUser.save();
    return done(null, newUser);
  } catch (err) {
    return done(err);
  }
});

const facebookStrategy = new FacebookStrategy({
  clientID: authConfig.facebook.clientId,
  clientSecret: authConfig.facebook.clientSecret,
  callbackURL: authConfig.facebook.callbackURL,
}, async (accessToken, refreshToken, profile, done) => {
  try {
    const user = await User.findOne({ facebookId: profile.id });
    if (user) {
      return done(null, user);
    }
    const newUser = new User({
      facebookId: profile.id,
      email: profile.emails[0].value,
      name: profile.displayName,
    });
    await newUser.save();
    return done(null, newUser);
  } catch (err) {
    return done(err);
  }
});

passport.use('local', localStrategy);
passport.use('google', googleStrategy);
passport.use('facebook', facebookStrategy);

router.post('/register', async (req, res) => {
  try {
    const { email, password, name } = req.body;
    const user = new User({ email, password, name });
    await user.save();
    res.json({ message: 'User created successfully' });
  } catch (err) {
    res.status(400).json({ message: 'Error creating user' });
  }
});

router.post('/login', passport.authenticate('local', { session: false }), (req, res) => {
  const user = req.user;
  const token = jwt.sign({ userId: user.id }, authConfig.secret, { expiresIn: '1h' });
  res.json({ token, user });
});

router.get('/google', passport.authenticate('google', { scope: ['profile', 'email'] }));

router.get('/google/callback', passport.authenticate('google', { failureRedirect: '/login' }), (req, res) => {
  const user = req.user;
  const token = jwt.sign({ userId: user.id }, authConfig.secret, { expiresIn: '1h' });
  res.json({ token, user });
});

router.get('/facebook', passport.authenticate('facebook', { scope: ['email'] }));

router.get('/facebook/callback', passport.authenticate('facebook', { failureRedirect: '/login' }), (req, res) => {
  const user = req.user;
  const token = jwt.sign({ userId: user.id }, authConfig.secret, { expiresIn: '1h' });
  res.json({ token, user });
});

router.get('/me', passport.authenticate('jwt', { session: false }), (req, res) => {
  const user = req.user;
  res.json({ user });
});

module.exports = router;
