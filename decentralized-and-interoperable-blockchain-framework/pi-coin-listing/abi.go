package pi_coin_listing

const PiCoinListingABI = `
[
    {
        "constant": true,
        "inputs": [],
        "name": "getPiCoinList",
        "outputs": [
            {
                "name": "",
                "type": "address[]"
            }
        ],
        "payable": false,
        "stateMutability": "view",
        "type": "function"
    },
    {
        "constant": false,
        "inputs": [
            {
                "name": "_piCoinAddress",
                "type": "address"
            }
        ],
        "name": "listPiCoin",
        "outputs": [],
        "payable": false,
        "stateMutability": "nonpayable",
        "type": "function"
    }
]
`
