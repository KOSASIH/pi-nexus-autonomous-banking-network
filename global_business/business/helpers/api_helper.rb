module ApiHelper
  # Returns a JSON response with a 200 OK status code
  def render_json(data, status = :ok)
    render json: data, status: status
  end

  # Returns a JSON response with a 404 Not Found status code
  def render_not_found
    render json: { error: 'Not Found' }, status: :not_found
  end

  # Returns a JSON response with a 401 Unauthorized status code
  def render_unauthorized
    render json: { error: 'Unauthorized' }, status: :unauthorized
  end

  # Returns a JSON response with a 500 Internal Server Error status code
  def render_error(error)
    render json: { error: error.message }, status: :internal_server_error
  end

  # Authenticates the API request using a secure token
  def authenticate_request
    token = request.headers['Authorization']
    return render_unauthorized unless token.present?

    begin
      payload, header = token.split('.')
      payload = JSON.parse(Base64.decode64(payload))
      user_id = payload['user_id']
      user = User.find_by(id: user_id)
      return render_unauthorized unless user.present?

      # Verify the token signature using a secure algorithm
      signature = OpenSSL::HMAC.hexdigest('sha256', Rails.application.secrets.secret_key_base, payload)
      return render_unauthorized unless signature == header

      # Set the current user
      @current_user = user
    rescue => e
      render_error(e)
    end
  end

  # Generates a secure token for API authentication
  def generate_token(user)
    payload = { user_id: user.id, exp: 1.hour.from_now.to_i }
    token = JSON.generate(payload)
    signature = OpenSSL::HMAC.hexdigest('sha256', Rails.application.secrets.secret_key_base, token)
    "#{Base64.encode64(token)}.#{signature}"
  end
end
