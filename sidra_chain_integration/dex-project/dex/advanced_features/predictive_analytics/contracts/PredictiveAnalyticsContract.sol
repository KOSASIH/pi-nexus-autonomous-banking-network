pragma solidity ^0.8.0;

contract PredictiveAnalyticsContract {
    mapping (address => mapping (address => uint256)) public predictions;

    function makePrediction(address _token, uint256 _prediction) public {
        require(_token!= address(0), "Invalid token address");
        predictions[msg.sender][_token] = _prediction;
    }

    function getPrediction(address _token) public view returns (uint256) {
        return predictions[msg.sender][_token];
    }
}
