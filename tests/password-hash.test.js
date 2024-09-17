// password-hash.test.js
const PasswordHash = require('./password-hash');

describe('PasswordHash', () => {
  it('should hash a password', async () => {
    const password = 'mysecretpassword';
    const hash = await PasswordHash.hash(password);
    expect(hash).not.toBe(password);
  });

  it('should compare a password with a hash', async () => {
    const password = 'mysecretpassword';
    const hash = await PasswordHash.hash(password);
    expect(await PasswordHash.compare(password, hash)).toBe(true);
  });
});
