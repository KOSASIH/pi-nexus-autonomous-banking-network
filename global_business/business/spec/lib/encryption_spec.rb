require 'rails_helper'

RSpec.describe Encryption do
  describe '#encrypt' do
    it 'encrypts the given data' do
      data = 'hello world'
      encrypted_data = Encryption.encrypt(data)
      expect(encrypted_data).not_to eq(data)
    end
  end

  describe '#decrypt' do
    it 'decrypts the given data' do
      data = 'hello world'
      encrypted_data = Encryption.encrypt(data)
      decrypted_data = Encryption.decrypt(encrypted_data)
      expect(decrypted_data).to eq(data)
    end
  end
end
