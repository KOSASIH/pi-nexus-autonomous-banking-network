// cypress/integration/api.spec.ts
describe('API', () => {
  it('returns a list of users', () => {
    cy.request('GET', 'https://example.com/api/users').then(response => {
      expect(response.body).to.be.an('array');
    });
  });

  it('creates a new user', () => {
    cy.request('POST', 'https://example.com/api/users', {
      name: 'John Doe',
      email: 'johndoe@example.com',
    }).then(response => {
      expect(response.body).to.have.property('id');
    });
  });
});
