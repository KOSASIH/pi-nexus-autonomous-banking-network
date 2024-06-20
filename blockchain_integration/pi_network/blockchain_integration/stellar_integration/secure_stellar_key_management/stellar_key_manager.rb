# stellar_key_manager.rb
require 'stellar-sdk'

class StellarKeyManager
  def initialize(stellar_secret_key, stellar_public_key)
    @stellar_secret_key = stellar_secret_key
    @stellar_public_key = stellar_public_key
  end

  def generate_keypair
    # Generate a new Stellar keypair
    keypair = Stellar::KeyPair.random
    return keypair.secret, keypair.public_key
  end

  def sign_transaction(transaction)
    # Sign a transaction with the secret key
    transaction.sign(@stellar_secret_key)
  end

  def verify_signature(transaction)
    # Verify the signature of a transaction
    transaction.verify(@stellar_public_key)
  end
end
