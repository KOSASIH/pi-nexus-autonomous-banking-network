require 'open-uri'
require 'json'

# Fiat currency exchange rate API endpoint
fiat_exchange_rate_api = 'https://api.exchangerate-api.com/v4/latest/'

# Set up API credentials
fiat_exchange_rate_api_key = 'YOUR_FIAT_EXCHANGE_RATE_API_KEY'

# Integrate banking system with fiat currency exchange rates
def integrate_fiat(banking_api, fiat_exchange_rate_api)
  # Make requests to fiat currency exchange rate API
  fiat_response = open-uri(fiat_exchange_rate_api + "?access_key=#{fiat_exchange_rate_api_key}").read
  fiat_exchange_rates = JSON.parse(fiat_response)['rates']

  # Use exchange rates to convert Pi coins to and from fiat currencies
  def convert_pi_to_fiat(pi_amount, fiat_currency)
    fiat_exchange_rate = fiat_exchange_rates[fiat_currency]
    fiat_amount = pi_amount * fiat_exchange_rate
    return fiat_amount
  end

  def convert_fiat_to_pi(fiat_amount, fiat_currency)
    fiat_exchange_rate = fiat_exchange_rates[fiat_currency]
    pi_amount = fiat_amount / fiat_exchange_rate
    return pi_amount
  end
end
