pragma solidity ^0.8.0;

contract AIContract {
    struct Model {
        string name;
        string description;
        string modelData;
        address owner;
        bool trained;
    }

    mapping(address => Model[]) private models;

    event ModelTrained(address indexed owner, uint256 indexed modelId, string name, string description);

    function trainModel(string memory _name, string memory _description, string memory _modelData) public {
        Model memory newModel;
        newModel.name = _name;
        newModel.description = _description;
        newModel.modelData = _modelData;
        newModel.owner = msg.sender;
        newModel.trained = false;

        uint256 modelId = models[msg.sender].length;
        models[msg.sender].push(newModel);

        emit ModelTrained(msg.sender, modelId, _name, _description);
    }

    function getModel(address _owner, uint256 _modelId) public view returns (string memory, string memory, string memory) {
        Model storage model = models[_owner][_modelId];
        return (model.name, model.description, model.modelData);
    }

    function setModelTrained(address _owner, uint256 _modelId) public {
        Model storage model = models[_owner][_modelId];
        model.trained = true;
    }
}
