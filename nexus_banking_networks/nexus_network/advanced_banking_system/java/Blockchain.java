// Blockchain.java
import java.security.MessageDigest;
import java.security.NoSuchAlgorithmException;
import java.util.ArrayList;
import java.util.List;

public class Blockchain {
    private List<Block> chain;
    private int difficulty;

    public Blockchain(int difficulty) {
        this.chain = new ArrayList<>();
        this.difficulty = difficulty;
        this.chain.add(createGenesisBlock());
    }

    private Block createGenesisBlock() {
        // Create the genesis block of the blockchain
        return new Block("Genesis Block", "0");
    }

    public Block getLatestBlock() {
        // Get the latest block in the blockchain
        return chain.get(chain.size() - 1);
    }

    public void addBlock(String data) {
        // Add a new block to the blockchain
        Block newBlock = new Block(data, getLatestBlock().getHash());
        newBlock.mineBlock(difficulty);
        chain.add(newBlock);
    }
}

class Block {
    private String data;
    private String previousHash;
    private String hash;

    public Block(String data, String previousHash) {
        this.data = data;
        this.previousHash = previousHash;
        this.hash = calculateHash();
    }

    private String calculateHash() {
        // Calculate the hash of the block
        try {
            MessageDigest md = MessageDigest.getInstance("SHA-256");
            String data = this.data + this.previousHash;
            byte[] bytes = md.digest(data.getBytes());
            StringBuilder sb = new StringBuilder();
            for (byte b : bytes) {
                sb.append(String.format("%02x", b));
            }
            return sb.toString();
        } catch (NoSuchAlgorithmException e) {
            throw new RuntimeException(e);
        }
    }

    public void mineBlock(int difficulty) {
        // Mine the block to find a valid hash
        while (!hash.substring(0, difficulty).equals(new String(new char[difficulty]).replace('\0', '0'))) {
            hash = calculateHash();
        }
    }
}
