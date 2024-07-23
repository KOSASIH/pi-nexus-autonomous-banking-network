import { shallow } from 'eact-testing-library';
import PiNetworkUI from './PiNetworkUI';

describe('PiNetworkUI', () => {
  it('renders the Pi Network UI component', () => {
    const wrapper = shallow(<PiNetworkUI />);
    expect(wrapper).toMatchSnapshot();
  });

  it('updates the from address state when the input value changes', () => {
    const wrapper = shallow(<PiNetworkUI />);
    const input = wrapper.find('input[type="text"]').first();
    input.simulate('change', { target: { value: '0x1234567890123456789012345678901234567890' } });
    expect(wrapper.state('fromAddress')).toEqual('0x1234567890123456789012345678901234567890');
  });

  it('updates the to address state when the input value changes', () => {
    const wrapper = shallow(<PiNetworkUI />);
    const input = wrapper.find('input[type="text"]').last();
    input.simulate('change', { target: { value: '0x1234567890123456789012345678901234567890' } });
    expect(wrapper.state('toAddress')).toEqual('0x1234567890123456789012345678901234567890');
  });

  it('updates the amount state when the input value changes', () => {
    const wrapper = shallow(<PiNetworkUI />);
    const input = wrapper.find('input[type="number"]');
    input.simulate('change', { target: { value: 100 } });
    expect(wrapper.state('amount')).toEqual(100);
  });

  it('submits the form when the submit button is clicked', () => {
    const handleSubmit = jest.fn();
    const wrapper = shallow(<PiNetworkUI handleSubmit={handleSubmit} />);
    const form = wrapper.find('form');
    form.simulate('submit', { preventDefault: () => {} });
    expect(handleSubmit).toHaveBeenCalled();
  });
});
