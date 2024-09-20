class AccessController
  def self.authorize(user, resource)
    if user.admin?
      true
    elsif user.role == resource.role
      true
    else
      false
    end
  end
end
