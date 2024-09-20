json.extract! @transaction, :id, :amount, :transaction_type, :created_at
json.account do
  json.extract! @transaction.account, :id, :account_number, :routing_number
end
json.user do
  json.extract! @transaction.account.user, :id, :username, :email
end
json.category do
  json.extract! @transaction.category, :id, :name
end
json.receipt_url @transaction.receipt_url
json.transaction_status @transaction.transaction_status
