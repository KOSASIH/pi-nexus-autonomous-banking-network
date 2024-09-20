require 'rails_helper'

RSpec.describe Api::V1::TransactionsController, type: :controller do
  describe 'GET #index' do
    it 'returns the transactions for the specified account' do
      user = FactoryBot.create(:user)
      account = FactoryBot.create(:account, user: user)
      transaction = FactoryBot.create(:transaction, account: account)
      sign_in user
      get :index, params: { account_id: account.id }, format: :json
      expect(response).to be_success
      expect(JSON.parse(response.body)).to eq([transaction.as_json])
    end
  end

  describe 'GET #show' do
    it 'returns the specified transaction' do
      user = FactoryBot.create(:user)
      account = FactoryBot.create(:account, user: user)
      transaction = FactoryBot.create(:transaction, account: account)
      sign_in user
      get :show, params: { account_id: account.id, id: transaction.id }, format: :json
      expect(response).to be_success
      expect(JSON.parse(response.body)).to eq(transaction.as_json)
    end
  end

  describe 'POST #create' do
    it 'creates a new transaction' do
      user = FactoryBot.create(:user)
      account = FactoryBot.create(:account, user: user)
      sign_in user
      post :create, params: { account_id: account.id, transaction: { amount: 100.0, type: 'deposit' } }, format: :json
      expect(response).to be_success
      expect(Transaction.count).to eq(1)
    end
  end

  describe 'PUT #update' do
    it 'updates the specified transaction' do
      user = FactoryBot.create(:user)
      account = FactoryBot.create(:account, user: user)
      transaction = FactoryBot.create(:transaction, account: account)
      sign_in user
      put :update, params: { account_id: account.id, id: transaction.id, transaction: { amount: 200.0 } }, format: :json
      expect(response).to be_success
      expect(transaction.reload.amount).to eq(200.0)
    end
  end

  describe 'DELETE #destroy' do
    it 'destroys the specified transaction' do
      user = FactoryBot.create(:user)
      account = FactoryBot.create(:account, user: user)
      transaction = FactoryBot.create(:transaction, account: account)
      sign_in user
      delete :destroy, params: { account_id: account.id, id: transaction.id }, format: :json
      expect(response).to be_success
      expect(Transaction.find_by(id: transaction.id)).to be_nil
    end
  end
end
