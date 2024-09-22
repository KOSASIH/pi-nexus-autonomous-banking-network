interface Transaction {
  from: string;
  to: string;
  value: number;
  data: string;
}

interface Message {
  type: string;
  payload: any;
}

class InteropProtocol {
  static async encodeTransaction(transaction: Transaction): Promise<string> {
    return JSON.stringify(transaction);
  }

  static async decodeTransaction(encodedTransaction: string): Promise<Transaction> {
    return JSON.parse(encodedTransaction);
  }

  static async encodeMessage(message: Message): Promise<string> {
    return JSON.stringify(message);
  }

  static async decodeMessage(encodedMessage: string): Promise<Message> {
    return JSON.parse(encodedMessage);
  }
}

export default InteropProtocol;
