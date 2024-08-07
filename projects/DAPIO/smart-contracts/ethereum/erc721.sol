pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC721/SafeERC721.sol";

contract ERC721 {
    using SafeERC721 for address;

    mapping (address => mapping (uint256 => ERC721Token)) public tokens;
    mapping (address => uint256) public balances;

    struct ERC721Token {
        address owner;
        uint256 tokenId;
        string tokenURI;
    }

    event Transfer(address indexed _from, address indexed _to, uint256 _tokenId);
    event Approval(address indexed _owner, address indexed _approved, uint256 _tokenId);
    event ApprovalForAll(address indexed _owner, address indexed _operator, bool _approved);

    function mint(address _to, uint256 _tokenId, string memory _tokenURI) public {
        require(tokens[_to][_tokenId].owner == address(0), "Token already exists");
        tokens[_to][_tokenId] = ERC721Token(_to, _tokenId, _tokenURI);
        balances[_to]++;
        emit Transfer(address(0), _to, _tokenId);
    }

    function transfer(address _to, uint256 _tokenId) public {
        require(tokens[msg.sender][_tokenId].owner == msg.sender, "Only the owner can transfer");
        require(tokens[_to][_tokenId].owner == address(0), "Token already exists");
        tokens[_to][_tokenId] = tokens[msg.sender][_tokenId];
        delete tokens[msg.sender][_tokenId];
        balances[msg.sender]--;
        balances[_to]++;
        emit Transfer(msg.sender, _to, _tokenId);
    }

    function approve(address _approved, uint256 _tokenId) public {
        require(tokens[msg.sender][_tokenId].owner == msg.sender, "Only the owner can approve");
        tokens[msg.sender][_tokenId].approved = _approved;
        emit Approval(msg.sender, _approved, _tokenId);
    }

    function getApproved(uint256 _tokenId) public view returns (address) {
        return tokens[msg.sender][_tokenId].approved;
    }

    function isApprovedForAll(address _owner, address _operator) public view returns (bool) {
        return tokens[_owner][_tokenId].approved == _operator;
    }
}
