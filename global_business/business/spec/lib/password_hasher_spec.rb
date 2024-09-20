require 'rails_helper'

RSpec.describe PasswordHasher do
  describe '#hash' do
    it 'hashes the given password' do
      password = 'hello world'
      hashed_password = PasswordHasher.hash(password)
      expect(hashed_password).not_to eq(password)
    end
  end

  describe '#verify' do
    it 'verifies the given password against the hashed password' do
      password = 'hello world'
      hashed_password = PasswordHasher.hash(password)
      expect(PasswordHasher.verify(password, hashed_password)).to be_truthy
    end
  end
end
