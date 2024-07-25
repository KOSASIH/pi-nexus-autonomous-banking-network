pragma solidity ^0.8.0;

contract Reputation {
    struct Review {
        address reviewer;
        uint256 rating;
        string comment;
    }

    mapping(address => mapping(bytes32 => Review[])) public reviews;

    function addReview(address _user, bytes32 _assetId, uint256 _rating, string memory _comment) public {
        Review memory review = Review(_user, _rating, _comment);
        reviews[_user][_assetId].push(review);
    }

    function getAverageRating(address _user, bytes32 _assetId) public view returns (uint256) {
        uint256 sum = 0;
        uint256 count = 0;
        for (uint256 i = 0; i < reviews[_user][_assetId].length; i++) {
            sum += reviews[_user][_assetId][i].rating;
            count++;
        }
        if (count == 0) {
            return 0;
        }
        return sum / count;
    }

    function getUserReputation(address _user) public view returns (uint256) {
        uint256 sum = 0;
        uint256 count = 0;
        for (bytes32 assetId in reviews[_user]) {
            sum += getAverageRating(_user, assetId);
            count++;
        }
        if (count == 0) {
            return 0;
        }
        return sum / count;
    }
}
