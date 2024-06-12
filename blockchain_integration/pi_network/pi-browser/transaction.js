import { hash } from 'hash-sum';
import { elliptic } from 'elliptic';

class Transaction {
  constructor(sender, recipient, amount) {
    this.sender = sender;
    this.recipient = recipient;
    this.amount = amount;
    this.timestamp = Date.now();
    this.hash = hash(this.toString());
  }

  toString() {
    return `${this.sender}:${this.recipient}:${this.amount}:${this.timestamp}`;
  }

  sign(privateKey) {
    const signature = elliptic.ec('secp256k1').sign(this.hash, privateKey);
    return signature;
  }
}

export default Transaction;
