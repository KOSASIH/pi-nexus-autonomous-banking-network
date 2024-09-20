require 'rails_helper'

RSpec.describe ApiHelper do
  describe '#json_response' do
    it 'renders a JSON response with the given object' do
      object = { foo: 'bar' }
      expect(json_response(object)).to eq(object.to_json)
    end
  end

  describe '#error_response' do
    it 'renders an error response with the given errors' do
      errors = ['error1', 'error2']
      expect(error_response(errors)).to eq({ errors: errors }.to_json)
    end
  end
end
