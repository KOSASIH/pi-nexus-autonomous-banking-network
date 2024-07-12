require "bcrypt"

class Security
  def self.hash_password(password)
    BCrypt::Password.create(password)
  end

  def self.verify_password(password, hashed_password)
    BCrypt::Password.new(hashed_password) == password
  end
end
