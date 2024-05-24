class DEXUtils {
  static calculatePrice (amount, reserve) {
    return (amount * 100) / reserve
  }

  static calculateReserve (amount, price) {
    return (amount * price) / 100
  }

  static calculateFee (amount, fee) {
    return (amount * fee) / 100
  }
}

module.exports = DEXUtils
