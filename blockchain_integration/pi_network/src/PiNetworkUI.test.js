import React from 'eact';
import { shallow } from 'eact-testing-library';
import PiNetworkUI from './PiNetworkUI';

describe('PiNetworkUI', () => {
  it('renders the Pi Network UI component', () => {
    const wrapper = shallow(<PiNetworkUI />);
    expect(wrapper).toMatchSnapshot();
  });
});
