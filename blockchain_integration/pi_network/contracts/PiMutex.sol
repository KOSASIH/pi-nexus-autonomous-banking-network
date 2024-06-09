pragma solidity ^0.8.0;

contract PiMutex {
    bool public locked;

    modifier lock() {
        require(!locked, "Mutex is already locked");
        locked = true;
        _;
        locked = false;
    }
}
