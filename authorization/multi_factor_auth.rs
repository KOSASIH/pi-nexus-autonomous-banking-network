// multi_factor_auth.rs
use rand::Rng;
use hmac::{Hmac, Mac};
use sha2::{Sha256, Digest};

struct MultiFactorAuth {
    user_id: String,
    password: String,
    otp_secret: String,
}

impl MultiFactorAuth {
    fn new(user_id: String, password: String, otp_secret: String) -> Self {
        MultiFactorAuth {
            user_id,
            password,
            otp_secret,
        }
    }

    fn authenticate(&self, input_password: &str, input_otp: &str) -> bool {
        // Verify password using HMAC
        let mut hmac = Hmac::<Sha256>::new(self.password.as_bytes());
        hmac.update(input_password.as_bytes());
        let password_mac = hmac.finalize();
        if password_mac!= self.password.as_bytes() {
            return false;
        }

        // Verify OTP using TOTP algorithm
        let totp = totp::Totp::new(self.otp_secret.as_bytes(), 30);
        let otp_mac = totp.generate(input_otp);
        if otp_mac!= input_otp {
            return false;
        }

        true
    }
}
