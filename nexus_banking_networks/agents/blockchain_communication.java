import org.web3j.protocol.Web3j;
import org.web3j.protocol.core.methods.response.TransactionReceipt;
import org.web3j.protocol.core.methods.response.Web3ClientVersion;
import org.web3j.protocol.http.HttpService;

public class BlockchainCommunication {
    private Web3j web3j;
    private String contractAddress;

    public BlockchainCommunication(String contractAddress, String nodeUrl) {
        this.contractAddress = contractAddress;
        this.web3j = Web3j.build(new HttpService(nodeUrl));
    }

    public void sendMessage(String message) {
        // Create a new transaction
        Transaction transaction = new Transaction(
                "0x" + contractAddress,
                "0x" + message,
                200000,
                20000
        );

        // Sign the transaction
        Credentials credentials = Credentials.create("0x1234567890abcdef");
        transaction.sign(credentials);

        // Send the transaction
        TransactionReceipt receipt = web3j.getTransactionReceipt(transaction).send();
        System.out.println("Transaction hash: " + receipt.getTransactionHash());
    }

    public String receiveMessage() {
        // Get the latest block
        Web3ClientVersion version = web3j.web3ClientVersion().send();
        String latestBlockHash = version.getWeb3ClientVersion();

        // Get the transaction from the latest block
        Transaction transaction = web3j.getTransactionByHash(latestBlockHash).send();

        // Decode the message
        String message = transaction.getInput();
        return message;
    }
}

// Example usage:
BlockchainCommunication communication = new BlockchainCommunication("0x1234567890abcdef", "https://mainnet.infura.io/v3/YOUR_PROJECT_ID");
communication.sendMessage("Hello, world!");
String receivedMessage = communication.receiveMessage();
System.out.println("Received message: " + receivedMessage);
