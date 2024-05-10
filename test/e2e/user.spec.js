// test/e2e/user.spec.js

describe('User Feature', () => {
  beforeEach(() => {
    cy.visit('/')
  })

  it('should register a new user', () => {
    cy.get('[data-testid="register-button"]').click()
    cy.get('[data-testid="first-name-input"]').type('John')
    cy.get('[data-testid="last-name-input"]').type('Doe')
    cy.get('[data-testid="email-input"]').type('john.doe@example.com')
    cy.get('[data-testid="password-input"]').type('password123')
    cy.get('[data-testid="register-form-button"]').click()
    cy.get('[data-testid="welcome-message"]').should(
      'contain',
      'Welcome, John Doe'
    )
  })

  it('should login a user', () => {
    cy.get('[data-testid="login-button"]').click()
    cy.get('[data-testid="email-input"]').type('john.doe@example.com')
    cy.get('[data-testid="password-input"]').type('password123')
    cy.get('[data-testid="login-form-button"]').click()
    cy.get('[data-testid="welcome-message"]').should(
      'contain',
      'Welcome, John Doe'
    )
  })
})
