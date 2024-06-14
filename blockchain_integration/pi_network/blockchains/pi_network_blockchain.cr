# pi_network_blockchain.cr
class PiNetworkBlockchain
  property chain : Array(Block)
  property pending_transactions : Array(Transaction)
  property miner_address : String

  def initialize
    @chain = [] of Block
    @pending_transactions = [] of Transaction
    @miner_address = ""
  end

  def create_new_block(miner_address : String) : Block
    # Create new block and add to chain
  end

  def add_transaction(tx : Transaction) : Nil
    # Add transaction to pending transactions
  end

  def mine_pending_transactions(miner_address : String) : Nil
    # Mine pending transactions and create new block
  end

  def get_balance_of_address(address : String) : Float64
    # Return balance of address
  end

  def get_transaction_by_id(tx_id : String) : Transaction?
    # Return transaction by ID
  end
end

class Block
  property index : Int32
  property timestamp : Time
  property transactions : Array(Transaction)
  property hash : String
  property prev_hash : String

  def initialize(@index, @timestamp, @transactions, @hash, @prev_hash)
  end
end

class Transaction
  property sender : String
  property recipient : String
  property amount : Float64

  def initialize(@sender, @recipient, @amount)
  end
end
