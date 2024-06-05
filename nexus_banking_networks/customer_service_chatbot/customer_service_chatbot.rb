# customer_service_chatbot.rb
require 'rubygems'
require 'nokogiri'
require 'open-uri'
require 'ai4r'

class CustomerServiceChatbot
  def initialize
    @ai = Ai4r::BayesClassifier.new
    train_chatbot
  end

  def train_chatbot
    # Train the chatbot using a dataset of customer queries and responses
    dataset = [
      ['What is my account balance?', 'Your account balance is $1000.'],
      ['How do I transfer money?', 'To transfer money, log in to your online banking account and follow the transfer instructions.'],
      # ...
    ]
    dataset.each do |query, response|
      @ai.add_example(query, response)
    end
  end

  def respond_to_query(query)
    # Use the trained chatbot to respond to customer queries
    response = @ai.classify(query)
    return response
  end
end

# Example usage:
chatbot = CustomerServiceChatbot.new
query = 'What is my account balance?'
response = chatbot.respond_to_query(query)
puts response
