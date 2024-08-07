pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/test/ERC721Test.sol";
import "../smart-contracts/ethereum/erc721.sol";

contract ERC721Test is ERC721Test {
    ERC721 private erc721;

    function setUp() public {
        erc721 = new ERC721();
    }

    function testMint() public {
        address alice = address(0x123);
        uint256 tokenId = 1;
        string memory tokenURI = "https://example.com/token/1";

        erc721.mint(alice, tokenId, tokenURI);
        assertEq(erc721.ownerOf(tokenId), alice);
    }

    function testTransfer() public {
        address alice = address(0x123);
        address bob = address(0x456);
        uint256 tokenId = 1;

        erc721.mint(alice, tokenId, "");
        erc721.transfer(bob, tokenId);
        assertEq(erc721.ownerOf(tokenId), bob);
    }

    function testApprove() public {
        address alice = address(0x123);
        address bob = address(0x456);
        uint256 tokenId = 1;

        erc721.mint(alice, tokenId, "");
        erc721.approve(bob, tokenId);
        assertEq(erc721.getApproved(tokenId), bob);
    }
}
