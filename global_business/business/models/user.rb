class User < ApplicationRecord
  # Encryption and decryption of sensitive data
  encrypts :ssn, :address, :phone_number

  # Secure password storage and verification
  has_secure_password validations: false
  validates :password, presence: true, length: { minimum: 12 }

  # Role-based access control
  enum role: { customer: 0, admin: 1 }

  # Associations
  has_many :accounts, dependent: :destroy
  has_many :transactions, through: :accounts

  # Validations
  validates :username, presence: true, uniqueness: true, length: { minimum: 3, maximum: 32 }
  validates :email, presence: true, uniqueness: true, format: { with: URI::MailTo::EMAIL_REGEXP }
  validates :first_name, presence: true, length: { minimum: 2, maximum: 50 }
  validates :last_name, presence: true, length: { minimum: 2, maximum: 50 }

  # Callbacks
  before_save :hash_ssn
  after_create :send_welcome_email

  private

  def hash_ssn
    self.ssn = Encryption.hash(self.ssn)
  end

  def send_welcome_email
    UserMailer.welcome_email(self).deliver_now
  end
end
