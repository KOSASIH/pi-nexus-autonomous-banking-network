export function calculateVolume(weight, price) {
  return weight * price;
}

export function calculateValue(volume, price) {
  return volume * price;
}

export function calculateFee(amount, feeRate) {
  return amount * feeRate;
}
