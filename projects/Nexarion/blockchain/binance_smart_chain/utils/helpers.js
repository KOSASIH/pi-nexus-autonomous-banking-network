export function toWei(value) {
  return web3.utils.toWei(value, 'ether');
}

export function fromWei(value) {
  return web3.utils.fromWei(value, 'ether');
}
