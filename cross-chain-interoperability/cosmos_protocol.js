const { CosmWasmClient } = require('@cosmjs/cosmwasm-stargate')
const { DirectSecp256k1HdWallet } = require('@cosmjs/proto-signing')
const { GasPrice } = require('@cosmjs/stargate')

const rpcEndpoint = 'https://cosmos-rpc.example.com'
const chainId = 'cosmos-testnet-123'
const wallet = await DirectSecp256k1HdWallet.fromMnemonic(
  'your-mnemonic-phrase'
)
const account = await wallet.getAccount('your-account-address')
const client = new CosmWasmClient(rpcEndpoint, {
  gasPrice: GasPrice.fromString('0.025uatom'),
  signer: wallet
})

const sendIbcTransfer = async (
  sourcePort,
  sourceChannel,
  destinationPort,
  destinationChannel,
  amount,
  sender
) => {
  const transferMsg = {
    type: 'cosmos-sdk/IBCTransfer',
    value: {
      source_port: sourcePort,
      source_channel: sourceChannel,
      token: {
        denom: 'uatom',
        amount: amount.toString()
      },
      sender,
      receiver: 'destination-address',
      timeout_height: '0',
      timeout_timestamp: '0'
    }
  }

  const tx = await client.signAndBroadcast(account, [transferMsg], 'auto')
  console.log('Transaction hash:', tx.transactionHash)
}

// Example usage
const sourcePort = 'transfer'
const sourceChannel = 'channel-1'
const destinationPort = 'transfer'
const destinationChannel = 'channel-2'
const amount = new BigNumber(1000000) // 1 uatom
const sender = 'your-account-address'

sendIbcTransfer(
  sourcePort,
  sourceChannel,
  destinationPort,
  destinationChannel,
  amount,
  sender
)
