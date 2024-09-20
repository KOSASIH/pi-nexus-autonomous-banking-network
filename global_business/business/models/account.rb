class Account < ApplicationRecord
  # Encryption and decryption of sensitive data
  encrypts :account_number, :routing_number

  # Associations
  belongs_to :user
  has_many :transactions, dependent: :destroy

  # Validations
  validates :account_number, presence: true, uniqueness: true, length: { minimum: 10, maximum: 20 }
  validates :routing_number, presence: true, uniqueness: true, length: { minimum: 9, maximum: 12 }
  validates :account_type, presence: true, inclusion: { in: %w[checking savings] }

  # Callbacks
  before_save :hash_account_number
  after_create :send_account_creation_email

  private

  def hash_account_number
    self.account_number = Encryption.hash(self.account_number)
  end

  def send_account_creation_email
    AccountMailer.account_creation_email(self).deliver_now
  end
end
