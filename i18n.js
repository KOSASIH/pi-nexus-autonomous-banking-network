const i18n = require('i18n');

i18n.configure({
  locales: ['en', 'es', 'fr', 'de'],
  defaultLocale: 'en',
  directory: './locales',
  indent: '  ',
  register: global,
});

module.exports = i18n;
