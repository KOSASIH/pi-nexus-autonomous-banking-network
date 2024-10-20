pragma solidity ^0.8.0;

import "https://github.com/OpenZeppelin/openzeppelin-solidity/contracts/token/ERC20/SafeERC20.sol";

contract BlockchainBasicsContract {
   // Mapping of user addresses to their progress
   mapping(address => uint) public progress;

   // Event emitted when a user completes a lesson
   event LessonCompleted(address indexed user, uint lessonId);

   // Event emitted when a user earns a badge
   event BadgeEarned(address indexed user, uint badgeId);

   // Struct to represent a lesson
   struct Lesson {
       uint id;
       string title;
       string description;
       uint reward;
   }

   // Struct to represent a badge
   struct Badge {
       uint id;
       string title;
       string description;
       uint requirement;
   }

   // Array of lessons
   Lesson[] public lessons;

   // Array of badges
   Badge[] public badges;

   // Function to complete a lesson
   function completeLesson(uint _lessonId) public {
       // Check if the user has already completed the lesson
       require(progress[msg.sender] < _lessonId, "Lesson already completed");

       // Update the user's progress
       progress[msg.sender] = _lessonId;

       // Emit the LessonCompleted event
       emit LessonCompleted(msg.sender, _lessonId);

       // Check if the user has earned a badge
       for (uint i = 0; i < badges.length; i++) {
           if (badges[i].requirement <= _lessonId) {
               // Emit the BadgeEarned event
               emit BadgeEarned(msg.sender, badges[i].id);
           }
       }
   }

   // Function to get a user's progress
   function getProgress(address _user) public view returns (uint) {
       return progress[_user];
   }
}
