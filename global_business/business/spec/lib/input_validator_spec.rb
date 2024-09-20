require 'rails_helper'

RSpec.describe InputValidator do
  describe '#validate' do
    it 'validates the given input' do
      input = 'hello world'
      expect(InputValidator.validate(input)).to be_truthy
    end
  end
end
