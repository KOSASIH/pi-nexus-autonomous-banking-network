class Node:
    def __init__(self, node_id, blockchain, difficulty_target):
        self.node_id = node_id
        self.blockchain = blockchain
        self.difficulty_target = difficulty_target
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.message_handler = MessageHandler(self)

    def start(self):
        self.socket.bind(("localhost", 8080 + self.node_id))
        self.socket.listen(5)
        print(f"Node {self.node_id} started")

        threading.Thread(target=self.listen_for_connections).start()
        threading.Thread(target=self.mine_blocks).start()

    def listen_for_connections(self):
        while True:
            conn, addr = self.socket.accept()
            print(f"Connected by {addr}")

            threading.Thread(target=self.handle_connection, args=(conn,)).start()

    def handle_connection(self, conn):
        while True:
            message = conn.recv(1024)
            if not message:
                break

            self.message_handler.handle_message(message)

    def mine_blocks(self):
        while True:
            block = self.blockchain.get_latest_block()
            transactions = self.blockchain.get_unconfirmed_transactions()
            new_block = Block(block.index + 1, block.hash, transactions, int(time.time()))
            miner = Miner(self.node_id, self.blockchain, self.difficulty_target)
            miner.mine(new_block)
            time.sleep(1)
