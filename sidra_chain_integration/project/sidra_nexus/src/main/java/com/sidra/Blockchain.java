package com.sidra;

import org.web3j.protocol.Web3j;
import org.web3j.protocol.core.methods.response.Web3ClientVersion;

public class Blockchain {
    public static void main(String[] args) {
        // Set up a Web3j instance
        Web3j web3j = Web3j.build(new HttpService("https://mainnet.infura.io/v3/YOUR_PROJECT_ID"));

        // Get the client version
        Web3ClientVersion clientVersion = web3j.getClientVersion().send();
        System.out.println(clientVersion.getWeb3ClientVersion());
    }
}
