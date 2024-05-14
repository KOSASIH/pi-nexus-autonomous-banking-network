const { ApiPromise, WsProvider } = require('@polkadot/api')
const { Keyring } = require('@polkadot/keyring')
const { encodeAddress } = require('@polkadot/util-crypto')

const wsProvider = new WsProvider('wss://polkadot.js.org/api')
const api = await ApiPromise.create({ provider: wsProvider })
const keyring = new Keyring({ type: 'sr25519' })

const polkadotAddress = encodeAddress(
  keyring.addFromUri('your-mnemonic-phrase').address,
  42
)

const sendCrossChainMessage = async (destinationChain, message) => {
  const {
    data: { Xcm: xcm }
  } = await api.tx.xcmPallet
    .send(destinationChain, message)
    .signAndSend(polkadotAddress)

  console.log('Cross-chain message sent:', xcm)
}

// Example usage
const destinationChain = 1234 // Replace with the ID of the destination chain
const message = 'Hello, cross-chain world!'

sendCrossChainMessage(destinationChain, message)
