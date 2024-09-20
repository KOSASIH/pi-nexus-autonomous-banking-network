require 'rails_helper'

RSpec.describe User, type: :model do
  it { is_expected.to validate_presence_of(:email) }
  it { is_expected.to validate_uniqueness_of(:email) }
  it { is_expected.to have_secure_password }
  it { is_expected.to have_many(:accounts) }

  describe 'password validation' do
    it 'validates password length' do
      user = User.new(password: 'short')
      expect(user).not_to be_valid
    end

    it 'validates password complexity' do
      user = User.new(password: 'password123')
      expect(user).not_to be_valid
    end
  end

  describe 'email validation' do
    it 'validates email format' do
      user = User.new(email: 'invalid_email')
      expect(user).not_to be_valid
    end
  end
end
