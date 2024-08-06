pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/SafeERC721.sol";

contract ERC721Token {
    string public name;
    string public symbol;
    uint public totalSupply;
    mapping (address => uint) public balances;
    mapping (uint => address) public tokenOwners;
    mapping (uint => string) public tokenURIs;

    constructor(string memory _name, string memory _symbol) public {
        name = _name;
        symbol = _symbol;
    }

    function mint(address _to, uint _tokenId, string memory _tokenURI) public {
        require(tokenOwners[_tokenId] == address(0), "Token already exists");
        tokenOwners[_tokenId] = _to;
        tokenURIs[_tokenId] = _tokenURI;
        balances[_to]++;
        totalSupply++;
        emit Transfer(address(0), _to, _tokenId);
    }

    function transfer(address _to, uint _tokenId) public {
        require(tokenOwners[_tokenId] == msg.sender, "Only the owner can transfer");
        tokenOwners[_tokenId] = _to;
        balances[msg.sender]--;
        balances[_to]++;
        emit Transfer(msg.sender, _to, _tokenId);
    }

    function ownerOf(uint _tokenId) public view returns (address) {
        return tokenOwners[_tokenId];
    }

    function tokenURI(uint _tokenId) public view returns (string memory) {
        return tokenURIs[_tokenId];
    }

    event Transfer(address indexed _from, address indexed _to, uint _tokenId);
}
