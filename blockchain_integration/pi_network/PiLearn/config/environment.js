module.exports = {
  development: {
    database: {
      uri: 'mongodb://localhost:27017/pilearn-dev',
      username: 'root',
      password: 'password'
    },
    server: {
      port: 3001,
      host: 'localhost'
    }
  },
  production: {
    database: {
      uri: 'mongodb://localhost:27017/pilearn-prod',
      username: 'root',
      password: 'password'
    },
    server: {
      port: 3000,
      host: 'localhost'
    }
  }
};
