pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/test/ERC20Test.sol";
import "../smart-contracts/pi-network/pi_token.sol";

contract PiTokenTest is ERC20Test {
    PiToken private piToken;

    function setUp() public {
        piToken = new PiToken();
    }

    function testTransfer() public {
        address alice = address(0x123);
        address bob = address(0x456);
        uint256 amount = 100;

        piToken.transfer(alice, amount);
        assertEq(piToken.balanceOf(alice), amount);

        piToken.transfer(bob, amount);
        assertEq(piToken.balanceOf(bob), amount);
    }

    function testApprove() public {
        address alice = address(0x123);
        address bob = address(0x456);
        uint256 amount = 100;

        piToken.approve(alice, amount);
        assertEq(piToken.allowance(alice, bob), amount);
    }

    function testTransferFrom() public {
        address alice = address(0x123);
        address bob = address(0x456);
        uint256 amount = 100;

        piToken.transfer(alice, amount);
        piToken.approve(alice, amount);
        piToken.transferFrom(alice, bob, amount);
        assertEq(piToken.balanceOf(bob), amount);
    }
}
