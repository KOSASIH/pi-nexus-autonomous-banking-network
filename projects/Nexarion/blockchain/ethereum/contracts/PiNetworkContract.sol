pragma solidity ^0.8.0;

contract PiNetworkContract {
    address[] public nodes;
    mapping (address => uint256) public piBalances;

    event NodeAdded(address indexed node);
    event NodeRemoved(address indexed node);
    event PiBalanceUpdated(address indexed user, uint256 balance);

    function addNode(address node) public {
        nodes.push(node);
        emit NodeAdded(node);
    }

    function removeNode(address node) public {
        for (uint256 i = 0; i < nodes.length; i++) {
            if (nodes[i] == node) {
                nodes[i] = nodes[nodes.length - 1];
                nodes.pop();
                emit NodeRemoved(node);
                return;
            }
        }
    }

    function updatePiBalance(address user, uint256 balance) public {
        piBalances[user] = balance;
        emit PiBalanceUpdated(user, balance);
    }

    function getPiBalance(address user) public view returns (uint256) {
        return piBalances[user];
    }

    function getNodeCount() public view returns (uint256) {
        return nodes.length;
    }

    function getNodeByAddress(address node) public view returns (bool) {
        for (uint256 i = 0; i < nodes.length; i++) {
            if (nodes[i] == node) {
                return true;
            }
        }
        return false;
    }
}
