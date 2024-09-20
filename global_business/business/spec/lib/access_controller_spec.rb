require 'rails_helper'

RSpec.describe AccessController do
  describe '#authorize' do
    it 'authorizes the given user to perform the given action' do
      user = FactoryBot.create(:user)
      action = 'read'
      expect(AccessController.authorize(user, action)).to be_truthy
    end
  end
end
