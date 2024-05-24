const express = require('express');
const app = express();
const i18n = require('./i18n');

app.use(i18n.init);

app.get('/', (req, res) => {
  res.render('index', { title: i18n.__('welcome') });
});

app.get('/login', (req, res) => {
  res.render('login', { title: i18n.__('login') });
});

app.post('/login', (req, res) => {
  // Login logic here
  res.redirect('/');
});

app.get('/register', (req, res) => {
  res.render('register', { title: i18n.__('register') });
});

app.post('/register', (req, res) => {
  // Register logic here
  res.redirect('/');
});

app.use((req, res, next) => {
  res.locals.lang = req.lang;
  next();
});

app.listen(3000, () => {
  console.log('Server started on port 3000');
});
