export function capitalizeFirstLetter(str) {
  return str.charAt(0).toUpperCase() + str.slice(1);
}

export function formatCurrency(amount, currency) {
  return `${amount.toFixed(2)} ${currency}`;
}
