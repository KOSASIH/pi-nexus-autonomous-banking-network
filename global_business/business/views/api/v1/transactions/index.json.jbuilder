json.array! @transactions do |transaction|
  json.extract! transaction, :id, :amount, :transaction_type, :created_at
  json.account do
    json.extract! transaction.account, :id, :account_number, :routing_number
  end
  json.user do
    json.extract! transaction.account.user, :id, :username, :email
  end
  json.url api_v1_transaction_url(transaction, format: :json)
  json.category do
    json.extract! transaction.category, :id, :name
  end
end
