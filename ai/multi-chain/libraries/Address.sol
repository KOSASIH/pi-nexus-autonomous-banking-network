pragma solidity ^0.8.0;

library Address {
    function isContract(address account) internal view returns (bool) {
        uint256 size;
        assembly {
            size := extcodesize(account)
        }
        return size > 0;
    }

    function sendValue(address payable recipient, uint256 amount) internal {
        require(recipient.send(amount), "Address: send value failed");
    }

    function transferValue(address payable recipient, uint256 amount) internal {
        require(recipient.transfer(amount), "Address: transfer value failed");
    }

    function call(address target, uint256 value, bytes memory data) internal returns (bytes memory) {
        require(address(this).balance >= value, "Address: insufficient balance for call");
        (bool success, bytes memory returndata) = target.call{value: value}(data);
        require(success, "Address: call failed");
        return returndata;
    }

    function delegateCall(address target, bytes memory data) internal returns (bytes memory) {
        (bool success, bytes memory returndata) = target.delegatecall(data);
        require(success, "Address: delegatecall failed");
        return returndata;
    }

    function staticCall(address target, bytes memory data) internal view returns (bytes memory) {
        (bool success, bytes memory returndata) = target.staticcall(data);
        require(success, "Address: staticcall failed");
        return returndata;
    }
}
