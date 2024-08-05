export function toWei(amount, unit) {
  return Web3.utils.toWei(amount, unit);
}

export function fromWei(amount, unit) {
  return Web3.utils.fromWei(amount, unit);
}

export function getGasPrice() {
  return Web3.utils.toWei('20', 'gwei');
}

export function getGasLimit() {
  return '2000000';
}
