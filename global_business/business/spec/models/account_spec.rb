require 'rails_helper'

RSpec.describe Account, type: :model do
  it { is_expected.to belong_to(:user) }
  it { is_expected.to have_many(:transactions) }
  it { is_expected.to validate_presence_of(:balance) }

  describe 'balance validation' do
    it 'validates balance format' do
      account = Account.new(balance: 'invalid_balance')
      expect(account).not_to be_valid
    end
  end
end
