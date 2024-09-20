json.extract! @user, :id, :username, :email, :first_name, :last_name, :role
json.accounts @user.accounts do |account|
  json.extract! account, :id, :account_number, :routing_number, :account_type
  json.url api_v1_account_url(account, format: :json)
end
json.transactions @user.transactions do |transaction|
  json.extract! transaction, :id, :amount, :transaction_type, :account_id
  json.url api_v1_transaction_url(transaction, format: :json)
end
