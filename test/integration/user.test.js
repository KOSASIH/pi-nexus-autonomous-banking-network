// test/integration/user.test.js

const chai = require('chai')
const chaiHttp = require('chai-http')
const app = require('../app')

const expect = chai.expect
chai.use(chaiHttp)

describe('User API Integration Tests', () => {
  describe('POST /register', () => {
    it('should create a new user and return a JWT', async () => {
      const res = await chai.request(app).post('/register').send({
        firstName: 'John',
        lastName: 'Doe',
        email: 'john.doe@example.com',
        password: 'password123'
      })

      expect(res).to.have.status(201)
      expect(res.body).to.be.an('object')
      expect(res.body).to.have.property('token')
      expect(res.body.token).to.be.a('string')
    })
  })

  describe('POST /login', () => {
    it('should login a user and return a JWT', async () => {
      const res = await chai.request(app).post('/login').send({
        email: 'john.doe@example.com',
        password: 'password123'
      })

      expect(res).to.have.status(200)
      expect(res.body).to.be.an('object')
      expect(res.body).to.have.property('token')
      expect(res.body.token).to.be.a('string')
    })
  })
})
