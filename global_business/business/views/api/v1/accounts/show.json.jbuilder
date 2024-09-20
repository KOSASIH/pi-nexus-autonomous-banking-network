json.extract! @account, :id, :account_number, :routing_number, :account_type
json.user do
  json.extract! @account.user, :id, :username, :email
end
json.transactions @account.transactions do |transaction|
  json.extract! transaction, :id, :amount, :transaction_type
  json.url api_v1_transaction_url(transaction, format: :json)
end
json.balance @account.balance
json.account_status @account.account_status
