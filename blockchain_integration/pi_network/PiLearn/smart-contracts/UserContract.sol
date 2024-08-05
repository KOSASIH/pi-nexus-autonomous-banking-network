pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/access/Roles.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/math/SafeMath.sol";
import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/utils/Address.sol";

contract UserContract {
    using Roles for address;
    using SafeMath for uint256;
    using Address for address;

    // Mapping of user addresses to user data
    mapping (address => UserData) public users;

    // Event emitted when a new user is created
    event NewUser(address indexed user, string name, string email);

    // Event emitted when a user is updated
    event UpdateUser(address indexed user, string name, string email);

    // Event emitted when a user is deleted
    event DeleteUser(address indexed user);

    // Struct to represent user data
    struct UserData {
        string name;
        string email;
        uint256 balance;
        uint256[] ownedCourses;
        uint256[] enrolledCourses;
        uint256[] createdCourses;
    }

    // Modifier to check if the caller is the owner of the contract
    modifier onlyOwner() {
        require(msg.sender == owner, "Only the owner can call this function");
        _;
    }

    // Modifier to check if the caller is a user
    modifier onlyUser(address user) {
        require(users[user].name != "", "Only users can call this function");
        _;
    }

    // Constructor function
    constructor() public {
        owner = msg.sender;
    }

    // Function to create a new user
    function createUser(string memory _name, string memory _email) public {
        require(users[msg.sender].name == "", "User already exists");
        users[msg.sender] = UserData(_name, _email, 0, new uint256[](0), new uint256[](0), new uint256[](0));
        emit NewUser(msg.sender, _name, _email);
    }

    // Function to update a user
    function updateUser(string memory _name, string memory _email) public onlyUser(msg.sender) {
        users[msg.sender].name = _name;
        users[msg.sender].email = _email;
        emit UpdateUser(msg.sender, _name, _email);
    }

    // Function to delete a user
    function deleteUser() public onlyUser(msg.sender) {
        delete users[msg.sender];
        emit DeleteUser(msg.sender);
    }

    // Function to get a user's data
    function getUserData(address user) public view returns (string memory, string memory, uint256, uint256[] memory, uint256[] memory, uint256[] memory) {
        return (users[user].name, users[user].email, users[user].balance, users[user].ownedCourses, users[user].enrolledCourses, users[user].createdCourses);
    }

    // Function to add a course to a user's owned courses
    function addOwnedCourse(address user, uint256 courseId) public onlyUser(user) {
        users[user].ownedCourses.push(courseId);
    }

    // Function to add a course to a user's enrolled courses
    function addEnrolledCourse(address user, uint256 courseId) public onlyUser(user) {
        users[user].enrolledCourses.push(courseId);
    }

    // Function to add a course to a user's created courses
    function addCreatedCourse(address user, uint256 courseId) public onlyUser(user) {
        users[user].createdCourses.push(courseId);
    }

    // Function to remove a course from a user's owned courses
    function removeOwnedCourse(address user, uint256 courseId) public onlyUser(user) {
        for (uint256 i = 0; i < users[user].ownedCourses.length; i++) {
            if (users[user].ownedCourses[i] == courseId) {
                users[user].ownedCourses[i] = users[user].ownedCourses[users[user].ownedCourses.length - 1];
                users[user].ownedCourses.pop();
                break;
            }
        }
    }

    // Function to remove a course from a user's enrolled courses
    function removeEnrolledCourse(address user, uint256 courseId) public onlyUser(user) {
        for (uint256 i = 0; i < users[user].enrolledCourses.length; i++) {
            if (users[user].enrolledCourses[i] == courseId) {
                users[user].enrolledCourses[i] = users[user].enrolledCourses[users[user].enrolledCourses.length - 1];
                users[user].enrolledCourses.pop();
                break;
            }
        }
    }

        // Function to remove a course from a user's created courses
    function removeCreatedCourse(address user, uint256 courseId) public onlyUser(user) {
        for (uint256 i = 0; i < users[user].createdCourses.length; i++) {
            if (users[user].createdCourses[i] == courseId) {
                users[user].createdCourses[i] = users[user].createdCourses[users[user].createdCourses.length - 1];
                users[user].createdCourses.pop();
                break;
            }
        }
    }

    // Function to get a user's balance
    function getBalance(address user) public view returns (uint256) {
        return users[user].balance;
    }

    // Function to add funds to a user's balance
    function addFunds(address user, uint256 amount) public onlyUser(user) {
        users[user].balance = users[user].balance.add(amount);
    }

    // Function to subtract funds from a user's balance
    function subtractFunds(address user, uint256 amount) public onlyUser(user) {
        require(users[user].balance >= amount, "Insufficient funds");
        users[user].balance = users[user].balance.sub(amount);
    }

    // Function to transfer funds from one user to another
    function transferFunds(address from, address to, uint256 amount) public onlyUser(from) {
        require(users[from].balance >= amount, "Insufficient funds");
        users[from].balance = users[from].balance.sub(amount);
        users[to].balance = users[to].balance.add(amount);
    }

    // Function to get a user's owned courses
    function getOwnedCourses(address user) public view returns (uint256[] memory) {
        return users[user].ownedCourses;
    }

    // Function to get a user's enrolled courses
    function getEnrolledCourses(address user) public view returns (uint256[] memory) {
        return users[user].enrolledCourses;
    }

    // Function to get a user's created courses
    function getCreatedCourses(address user) public view returns (uint256[] memory) {
        return users[user].createdCourses;
    }

    // Function to check if a user owns a course
    function ownsCourse(address user, uint256 courseId) public view returns (bool) {
        for (uint256 i = 0; i < users[user].ownedCourses.length; i++) {
            if (users[user].ownedCourses[i] == courseId) {
                return true;
            }
        }
        return false;
    }

    // Function to check if a user is enrolled in a course
    function isEnrolledInCourse(address user, uint256 courseId) public view returns (bool) {
        for (uint256 i = 0; i < users[user].enrolledCourses.length; i++) {
            if (users[user].enrolledCourses[i] == courseId) {
                return true;
            }
        }
        return false;
    }

    // Function to check if a user created a course
    function createdCourse(address user, uint256 courseId) public view returns (bool) {
        for (uint256 i = 0; i < users[user].createdCourses.length; i++) {
            if (users[user].createdCourses[i] == courseId) {
                return true;
            }
        }
        return false;
    }
}
