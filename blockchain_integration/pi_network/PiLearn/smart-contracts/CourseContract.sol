pragma solidity ^0.8.0;

contract CourseContract {
    address private owner;
    mapping (address => Course) public courses;

    struct Course {
        string title;
        string description;
        uint256 price;
    constructor() public {
        owner = msg.sender;
    }

    function createCourse(string memory _title, string memory _description, uint256 _price) public {
        Course memory newCourse = Course(_title, _description, _price);
        courses[msg.sender] = newCourse;
    }

    function getCourses() public view returns (Course[] memory) {
        Course[] memory courseList = new Course[](courses.length);
        for (uint256 i = 0; i < courses.length; i++) {
            courseList[i] = courses[i];
        }
        return courseList;
    }

    function getCourse(address _address) public view returns (Course memory) {
        return courses[_address];
    }
}   
