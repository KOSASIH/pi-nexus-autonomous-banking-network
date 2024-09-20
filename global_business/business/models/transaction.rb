class Transaction < ApplicationRecord
  # Encryption and decryption of sensitive data
  encrypts :amount

  # Associations
  belongs_to :account
  belongs_to :user

  # Validations
  validates :amount, presence: true, numericality: { greater_than: 0.01, less_than: 100000.00 }
  validates :transaction_type, presence: true, inclusion: { in: %w[deposit withdrawal] }

  # Callbacks
  before_save :hash_amount
  after_create :send_transaction_email

  private

  def hash_amount
    self.amount = Encryption.hash(self.amount)
  end

  def send_transaction_email
    TransactionMailer.transaction_email(self).deliver_now
  end
end
