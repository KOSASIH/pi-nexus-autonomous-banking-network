json.array! @users do |user|
  json.extract! user, :id, :username, :email, :first_name, :last_name, :role
  json.url api_v1_user_url(user, format: :json)
end
