require 'rails_helper'

RSpec.describe Api::V1::UsersController, type: :controller do
  describe 'GET #show' do
    it 'returns the current user' do
      user = FactoryBot.create(:user)
      sign_in user
      get :show, format: :json
      expect(response).to be_success
      expect(JSON.parse(response.body)).to eq(user.as_json)
    end
  end

  describe 'PUT #update' do
    it 'updates the current user' do
      user = FactoryBot.create(:user)
      sign_in user
      put :update, params: { user: { email: 'new_email@example.com' } }, format: :json
      expect(response).to be_success
      expect(user.reload.email).to eq('new_email@example.com')
    end
  end

  describe 'DELETE #destroy' do
    it 'destroys the current user' do
      user = FactoryBot.create(:user)
      sign_in user
      delete :destroy, format: :json
      expect(response).to be_success
      expect(User.find_by(id: user.id)).to be_nil
    end
  end
end
