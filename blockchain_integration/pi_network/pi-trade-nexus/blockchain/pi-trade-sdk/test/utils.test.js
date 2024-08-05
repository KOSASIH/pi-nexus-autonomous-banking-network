import { toWei, fromWei, getGasPrice, getGasLimit } from '../lib/utils';

describe('utils', () => {
  it('should convert to wei', () => {
    const amount = '10';
    const unit = 'ether';
    const weiAmount = toWei(amount, unit);
    expect(weiAmount).to.be.a('string');
  });

  it('should convert from wei', () => {
    const amount = '1000000000000000000';
    const unit = 'ether';
    const etherAmount = fromWei(amount, unit);
    expect(etherAmount).to.be.a('string');
  });

  it('should get gas price', () => {
    const gasPrice = getGasPrice();
    expect(gasPrice).to.be.a('string');
  });

  it('should get gas limit', () => {
    const gasLimit = getGasLimit();
    expect(gasLimit).to.be.a('string');
  });
});
