// test/user.test.js

const chai = require('chai');
const chaiHttp = require('chai-http');
const app = require('../app');
const User = require('../models/user');

const expect = chai.expect;
chai.use(chaiHttp);

describe('User API', () => {
  describe('POST /register', () => {
    it('should create a new user', async () => {
      const res = await chai.request(app).post('/register').send({
        firstName: 'John',
        lastName: 'Doe',
        email: 'john.doe@example.com',
        password: 'password123'
      });

      expect(res).to.have.status(201);
      expect(res.body).to.be.an('object');
      expect(res.body.user).to.have.property('_id');
      expect(res.body.user).to.have.property('firstName', 'John');
      expect(res.body.user).to.have.property('lastName', 'Doe');
      expect(res.body.user).to.have.property('email', 'john.doe@example.com');
      expect(res.body.user).to.have.property('password', 'password123');
    });
  });

  describe('POST /login', () => {
    it('should login a user', async () => {
      const user = new User({
        firstName: 'Jane',
        lastName: 'Doe',
        email: 'jane.doe@example.com',
        password: 'password123'
      });

      await user.save();

      const res = await chai.request(app).post('/login').send({
        email: 'jane.doe@example.com',
        password: 'password123'
      });

      expect(res).to.have.status(200);
      expect(res.body).to.be.an('object');
      expect(res.body.user).to.have.property('_id');
      expect(res.body.user).to.have.property('firstName', 'Jane');
      expect(res.body.user).to.have.property('lastName', 'Doe');
      expect(res.body.user).to.have.property('email', 'jane.doe@example.com');
      expect(res.body.user).to.have.property('password', 'password123');
    });
  });
});
