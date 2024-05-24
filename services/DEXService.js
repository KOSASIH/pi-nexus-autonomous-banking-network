const Web3 = require("web3");
const DEX = require("../contracts/DEX.json");

class DEXService {
  constructor(web3, contractAddress) {
    this.web3 = web3;
    this.contract = new this.web3.eth.Contract(DEX.abi, contractAddress);
  }

  // The function to create a new order
  createOrder(amount, price) {
    return new Promise((resolve, reject) => {
      this.contract.methods.createOrder(amount, price).send(
        {
          from: this.web3.eth.defaultAccount,
        },
        (error, result) => {
          if (error) {
            reject(error);
          } else {
            resolve(result);
          }
        },
      );
    });
  }

  // The function to cancel an order
  cancelOrder(orderId) {
    return new Promise((resolve, reject) => {
      this.contract.methods.cancelOrder(orderId).send(
        {
          from: this.web3.eth.defaultAccount,
        },
        (error, result) => {
          if (error) {
            reject(error);
          } else {
            resolve(result);
          }
        },
      );
    });
  }

  // The function to execute a trade
  executeTrade(trader, orderId) {
    return new Promise((resolve, reject) => {
      this.contract.methods.executeTrade(trader, orderId).send(
        {
          from: this.web3.eth.defaultAccount,
        },
        (error, result) => {
          if (error) {
            reject(error);
          } else {
            resolve(result);
          }
        },
      );
    });
  }

  // The function to listen for the 'TradeExecuted' event
  listenForTradeExecuted() {
    this.contract.events.TradeExecuted(
      {
        fromBlock: "latest",
      },
      (error, event) => {
        if (error) {
          console.log("Error: " + error);
        } else {
          console.log("Trade executed: " + JSON.stringify(event.returnValues));
        }
      },
    );
  }
}

module.exports = DEXService;
