// SPDX-License-Identifier: MIT
pragma solidity ^0.8.10;

import "truffle/files/openzeppelin/test/helpers/Assert.sol";
import "truffle/files/openzeppelin/test/helpers/ExpectedError.sol";
import "truffle/files/openzeppelin/test/helpers/SetupContext.sol";

import "../contracts/Token.sol";

contract TokenTest is SetupContext {
    Token public token;

    function setUp() public {
        token = new Token();
    }

    function testTotalSupply() public {
        Assert.equal(token.totalSupply(), 10000, "Token: total supply is incorrect");
    }

    function testBalanceOf() public {
        Assert.equal(token.balanceOf(address(this)), 1000, "Token: balance of is incorrect");
    }

    function testTransfer() public {
        token.transfer(address(this), 500);
        Assert.equal(token.balanceOf(address(this)), 1500, "Token: balance of is incorrect after transfer");
    }

    function testTransferFrom() public {
        token.transfer(address(this), 500);
        token.transferFrom(address(this), address(this), 250);
        Assert.equal(token.balanceOf(address(this)), 1250, "Token: balance of is incorrect after transferFrom");
    }

    function testTransferTo() public {
        token.transfer(address(this), 500);
        token.transferTo(address(this), 250);
        Assert.equal(token.balanceOf(address(this)), 750, "Token: balance of is incorrect after transferTo");
    }

    function testTransferWithValue() public {
        token.transfer(address(this), 500, 12345);
        Assert.equal(token.balanceOf(address(this)), 500, "Token: balance of is incorrect after transferWithValue");
    }

    function testTransferWithData() public {
        token.transfer(address(this), 500, "");
        Assert.equal(token.balanceOf(address(this)), 500, "Token: balance of is incorrect after transferWithData");
    }

    function testTransferWithGasLimit() public {
        token.transfer(address(this), 500, "", 100000);
        Assert.equal(token.balanceOf(address(this)), 500, "Token: balance of is incorrect after transferWithGasLimit");
    }

    function testTransferWithEta() public {
        token.transfer(address(this), 500, "", 100000, 12345);
        Assert.equal(token.balanceOf(address(this)), 500, "Token: balance of is incorrect after transferWithEta");
    }

    function testTransferWithNonce() public {
        token.transfer(address(this), 500, "", 100000, 12345, 67890);
        Assert.equal(token.balanceOf(address(this)), 500, "Token: balance of is incorrect after transferWithNonce");
    }

    function testTransferWithV() public {
        token.transfer(address(this), 500, "", 100000, 12345, 67890, 0x12345678);
        Assert.equal(token.balanceOf(address(this)), 500, "Token: balance of is incorrect after transferWithV");
    }
}
