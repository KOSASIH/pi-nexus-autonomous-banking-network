// cypress/integration/math.spec.ts
describe('Math API', () => {
  beforeEach(() => {
    cy.visit('https://example.com/math');
  });

  it('adds two numbers', () => {
    cy.get('input[name="a"]').type('2');
    cy.get('input[name="b"]').type('3');
    cy.get('button[type="submit"]').click();
    cy.get('p.result').should('contain', '5');
  });

  it('subtracts two numbers', () => {
    cy.get('input[name="a"]').type('5');
    cy.get('input[name="b"]').type('3');
    cy.get('button[type="submit"]').click();
    cy.get('p.result').should('contain', '2');
  });
});
