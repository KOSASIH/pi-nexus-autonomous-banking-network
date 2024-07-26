import BN from 'bn.js';

class BigNumber {
  static fromNumber(number) {
    return new BN(number);
  }

  static fromString(string) {
    return new BN(string, 10);
  }

  static fromHex(hex) {
    return new BN(hex, 16);
  }

  static add(a, b) {
    return a.add(b);
  }

  static sub(a, b) {
    return a.sub(b);
  }

  static mul(a, b) {
    return a.mul(b);
  }

  static div(a, b) {
    return a.div(b);
  }

  static mod(a, b) {
    return a.mod(b);
  }

  static eq(a, b) {
    return a.eq(b);
  }

  static lt(a, b) {
    return a.lt(b);
  }

  static lte(a, b) {
    return a.lte(b);
  }

  static gt(a, b) {
    return a.gt(b);
  }

  static gte(a, b) {
    return a.gte(b);
  }

  static toNumber(bigNumber) {
    return bigNumber.toNumber();
  }

  static toString(bigNumber) {
    return bigNumber.toString(10);
  }

  static toHex(bigNumber) {
    return bigNumber.toString(16);
  }

  static isZero(bigNumber) {
    return bigNumber.eq(0);
  }

  static isNegative(bigNumber) {
    return bigNumber.lt(0);
  }

  static isPositive(bigNumber) {
    return bigNumber.gt(0);
  }

  static abs(bigNumber) {
    return bigNumber.abs();
  }

  static sqrt(bigNumber) {
    return bigNumber.sqrt();
  }

  static pow(bigNumber, exponent) {
    return bigNumber.pow(exponent);
  }

  static gcd(a, b) {
    return a.gcd(b);
  }

  static lcm(a, b) {
    return a.mul(b).div(a.gcd(b));
  }
}

export default BigNumber;
