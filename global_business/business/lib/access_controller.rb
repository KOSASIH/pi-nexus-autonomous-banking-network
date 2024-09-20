class AccessController
  def self.authorize(user, resource)
    return false unless user.present? && resource.present?
    return false unless user.roles.include?(resource.role)

    # Verify permissions using a secure algorithm
    permission_token = generate_permission_token(user, resource)
    verify_permission_token(permission_token)
  end

  private

  def self.generate_permission_token(user, resource)
    # Generate a permission token using a secure algorithm
    OpenSSL::HMAC.hexdigest('sha256', Rails.application.secrets.secret_key_base, "#{user.id}#{resource.id}")
  end

  def self.verify_permission_token(permission_token)
    # Verify the permission token using a secure algorithm
    signature = OpenSSL::HMAC.hexdigest('sha256', Rails.application.secrets.secret_key_base, permission_token)
    signature == permission_token
  end
end
