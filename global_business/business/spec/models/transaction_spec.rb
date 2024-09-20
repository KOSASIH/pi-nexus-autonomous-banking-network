require 'rails_helper'

RSpec.describe Transaction, type: :model do
  it { is_expected.to belong_to(:account) }
  it { is_expected.to validate_presence_of(:amount) }
  it { is_expected.to validate_presence_of(:type) }

  describe 'amount validation' do
    it 'validates amount format' do
      transaction = Transaction.new(amount: 'invalid_amount')
      expect(transaction).not_to be_valid
    end
  end

  describe 'type validation' do
    it 'validates type format' do
      transaction = Transaction.new(type: 'invalid_type')
      expect(transaction).not_to be_valid
    end
  end
end
