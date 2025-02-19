import speakeasy from 'speakeasy';
import qrcode from 'qrcode';

export const generateMFASecret = () => {
    const secret = speakeasy.generateSecret({ length: 20 });
    return secret;
};

export const generateQRCode = async (secret, userEmail) => {
    const otpauth = `otpauth://totp/${userEmail}?secret=${secret.base32}&issuer=PiNexus`;
    const qrCode = await qrcode.toDataURL(otpauth);
    return qrCode;
};

export const verifyMFA = (token, secret) => {
    return speakeasy.totp.verify({
        secret: secret.base32,
        encoding: 'base32',
        token: token,
    });
};
