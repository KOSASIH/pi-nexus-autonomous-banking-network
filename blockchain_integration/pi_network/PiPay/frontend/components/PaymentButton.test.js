// frontend/components/PaymentButton.test.js
import React from 'react';
import { render, fireEvent, waitFor } from '@testing-library/react';
import { Web3Provider } from '../../contexts/Web3Context';
import { PaymentGatewayProvider } from '../../contexts/PaymentGatewayContext';
import { PaymentButton } from './PaymentButton';

describe('PaymentButton', () => {
  it('renders correctly', () => {
    const { getByText } = render(
      <Web3Provider>
        <PaymentGatewayProvider>
          <PaymentButton amount={1.0} payer="0x1234567890abcdef" payee="0xabcdef1234567890" />
        </PaymentGatewayProvider>
      </Web3Provider>
    );

    expect(getByText('Pay 1.0 ETH')).toBeInTheDocument();
  });

  it('calls payment function on click', async () => {
    const paymentFunction = jest.fn();
    const { getByText } = render(
      <Web3Provider>
        <PaymentGatewayProvider paymentFunction={paymentFunction}>
          <PaymentButton amount={1.0} payer="0x1234567890abcdef" payee="0xabcdef1234567890" />
        </PaymentGatewayProvider>
      </Web3Provider>
    );

    const button = getByText('Pay 1.0 ETH');
    fireEvent.click(button);

    await waitFor(() => expect(paymentFunction).toHaveBeenCalledTimes(1));
  });

  it('disables button when payment is in progress', async () => {
    const paymentFunction = jest.fn();
    const { getByText } = render(
      <Web3Provider>
        <PaymentGatewayProvider paymentFunction={paymentFunction}>
          <PaymentButton amount={1.0} payer="0x1234567890abcdef" payee="0xabcdef1234567890" />
        </PaymentGatewayProvider>
      </Web3Provider>
    );

    const button = getByText('Pay 1.0 ETH');
    fireEvent.click(button);

    expect(button).toBeDisabled();

    await waitFor(() => expect(paymentFunction).toHaveBeenCalledTimes(1));

    expect(button).not.toBeDisabled();
  });

  it('renders error message on payment failure', async () => {
    const paymentFunction = jest.fn(() => Promise.reject(new Error('Payment failed')));
    const { getByText } = render(
      <Web3Provider>
        <PaymentGatewayProvider paymentFunction={paymentFunction}>
          <PaymentButton amount={1.0} payer="0x1234567890abcdef" payee="0xabcdef1234567890" />
        </PaymentGatewayProvider>
      </Web3Provider>
    );

    const button = getByText('Pay 1.0 ETH');
    fireEvent.click(button);

    await waitFor(() => expect(getByText('Payment failed')).toBeInTheDocument());
  });
});
