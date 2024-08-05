import React from 'eact';
import { render, waitFor } from '@testing-library/react';
import { Launches } from '../client/components/Launches';
import { Provider } from 'eact-redux';
import { store } from '../client/store';

describe('Launches component', () => {
  it('renders launch list', async () => {
    const { getByText } = render(
      <Provider store={store}>
        <Launches />
      </Provider>
    );

    await waitFor(() => getByText('Launch 1'));
    expect(getByText('Launch 1')).toBeInTheDocument();
    expect(getByText('Launch 2')).toBeInTheDocument();
  });
});
