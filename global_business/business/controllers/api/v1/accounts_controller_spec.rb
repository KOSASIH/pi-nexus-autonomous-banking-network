require 'rails_helper'

RSpec.describe Api::V1::AccountsController, type: :controller do
  describe 'GET #index' do
    it 'returns the current user\'s accounts' do
      user = FactoryBot.create(:user)
      account = FactoryBot.create(:account, user: user)
      sign_in user
      get :index, format: :json
      expect(response).to be_success
      expect(JSON.parse(response.body)).to eq([account.as_json])
    end
  end

  describe 'GET #show' do
    it 'returns the specified account' do
      user = FactoryBot.create(:user)
      account = FactoryBot.create(:account, user: user)
      sign_in user
      get :show, params: { id: account.id }, format: : json
      expect(response).to be_success
      expect(JSON.parse(response.body)).to eq(account.as_json)
    end
  end

  describe 'POST #create' do
    it 'creates a new account' do
      user = FactoryBot.create(:user)
      sign_in user
      post :create, params: { account: { balance: 100.0 } }, format: :json
      expect(response).to be_success
      expect(Account.count).to eq(1)
    end
  end

  describe 'PUT #update' do
    it 'updates the specified account' do
      user = FactoryBot.create(:user)
      account = FactoryBot.create(:account, user: user)
      sign_in user
      put :update, params: { id: account.id, account: { balance: 200.0 } }, format: :json
      expect(response).to be_success
      expect(account.reload.balance).to eq(200.0)
    end
  end

  describe 'DELETE #destroy' do
    it 'destroys the specified account' do
      user = FactoryBot.create(:user)
      account = FactoryBot.create(:account, user: user)
      sign_in user
      delete :destroy, params: { id: account.id }, format: :json
      expect(response).to be_success
      expect(Account.find_by(id: account.id)).to be_nil
    end
  end
end
