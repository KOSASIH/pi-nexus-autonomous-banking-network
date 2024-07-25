package com.sidra.nexus;

import org.ethereum.core.Block;
import org.ethereum.core.Transaction;
import org.ethereum.facade.Ethereum;
import org.ethereum.facade.EthereumFactory;

public class BlockchainManager {
    private Ethereum ethereum;

    public BlockchainManager() {
        ethereum = EthereumFactory.createEthereum();
    }

    public void createTransaction(String from, String to, double amount) {
        Transaction transaction = ethereum.getTransaction(from, to, amount);
        ethereum.sendTransaction(transaction);
    }

    public Block getLatestBlock() {
        return ethereum.getBlockchain().getBestBlock();
    }

    public void mineBlock() {
        ethereum.getMining().mine();
    }
}
