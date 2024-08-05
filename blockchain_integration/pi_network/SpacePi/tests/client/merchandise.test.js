import React from 'eact';
import { render, waitFor } from '@testing-library/react';
import { Merchandise } from '../client/components/Merchandise';
import { Provider } from 'eact-redux';
import { store } from '../client/store';

describe('Merchandise component', () => {
  it('renders merchandise list', async () => {
    const { getByText } = render(
      <Provider store={store}>
        <Merchandise />
      </Provider>
    );

    await waitFor(() => getByText('T-Shirt'));
    expect(getByText('T-Shirt')).toBeInTheDocument();
    expect(getByText('Hat')).toBeInTheDocument();
  });
});
