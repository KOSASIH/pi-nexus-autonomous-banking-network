require 'openssl'
require 'base64'

class Encryption
  def self.encrypt(data, key)
    cipher = OpenSSL::Cipher.new('AES-256-CBC')
    cipher.encrypt
    cipher.key = key

    encrypted_data = cipher.update(data) + cipher.final
    Base64.encode64(encrypted_data)
  end

  def self.decrypt(encrypted_data, key)
    encrypted_data = Base64.decode64(encrypted_data)
    cipher = OpenSSL::Cipher.new('AES-256-CBC')
    cipher.decrypt
    cipher.key = key

    decrypted_data = cipher.update(encrypted_data) + cipher.final
    decrypted_data
  end
end
