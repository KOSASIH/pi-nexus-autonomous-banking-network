// src/integrations/paymentGateway.ts
import Stripe from 'stripe';

const stripe = new Stripe('your-stripe-secret-key');

export const createPaymentIntent = async (amount: number, currency: string) => {
    const paymentIntent = await stripe.paymentIntents.create({
        amount,
        currency,
    });
    return paymentIntent;
};
