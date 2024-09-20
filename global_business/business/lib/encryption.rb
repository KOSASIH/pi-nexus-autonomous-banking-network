require 'openssl'

class Encryption
  def self.encrypt(data)
    cipher = OpenSSL::Cipher.new('AES-256-GCM')
    cipher.encrypt
    cipher.key = Rails.application.secrets.secret_key_base
    encrypted_data = cipher.update(data) + cipher.final
    encrypted_data
  end

  def self.decrypt(encrypted_data)
    cipher = OpenSSL::Cipher.new('AES-256-GCM')
    cipher.decrypt
    cipher.key = Rails.application.secrets.secret_key_base
    decrypted_data = cipher.update(encrypted_data) + cipher.final
    decrypted_data
  end
end
