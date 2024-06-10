import pyotp

class MFAAuth:
    def __init__(self):
        self.totp = pyotp.TOTP("base32secretkey")

    def authenticate(self, otp):
        if self.totp.verify(otp):
            return True
        else:
            return False

# Example usage:
mfa_auth = MFAAuth()
otp = input("Enter OTP: ")
if mfa_auth.authenticate(otp):
    print("Authenticated successfully!")
else:
    print("Authentication failed!")
