import React, { useState } from 'react';
import { requestPasswordReset, resetPassword } from '../../api';

const PasswordReset = () => {
    const [email, setEmail] = useState('');
    const [token, setToken] = useState('');
    const [newPassword, setNewPassword] = useState('');
    const [step, setStep] = useState(1);
    const [message, setMessage] = useState('');

    const handleRequestReset = async (e) => {
        e.preventDefault();
        try {
            await requestPasswordReset(email);
            setMessage('Password reset link sent to your email.');
            setStep(2);
        } catch (err) {
            setMessage(err.response.data.message);
        }
    };

    const handleResetPassword = async (e) => {
        e.preventDefault();
        try {
            await resetPassword(token, newPassword);
            setMessage('Password has been reset successfully.');
        } catch (err) {
            setMessage(err.response.data.message);
        }
    };

    return (
        <div>
            {step === 1 ? (
                <form onSubmit={handleRequestReset}>
                    <h2>Request Password Reset</h2>
                    <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} placeholder="Email" required />
                    <button type="submit">Send Reset Link</button>
                </form>
            ) : (
                <form onSubmit={handleResetPassword}>
                    <h2>Reset Password</h2>
                    <input type="text" value={token} onChange={(e) => setToken(e.target.value)} placeholder="Reset Token" required />
                    <input type="password" value={newPassword} onChange={(e) => setNewPassword(e.target.value)} placeholder="New Password" required />
                    <button type="submit">Reset Password</button>
                </form>
            )}
            {message && <p>{message}</p>}
        </div>
    );
};

export default PasswordReset;
