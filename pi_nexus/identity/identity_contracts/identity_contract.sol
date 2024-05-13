pragma solidity ^0.8.0;

contract IdentityContract {
    struct Identity {
        string name;
        mapping(string => string) attributes;
    }

    mapping(address => Identity[]) private identities;

    event IdentityCreated(address indexed owner, uint256 indexed identityId, string name, string[] memory attributes);

    function createIdentity(string memory _name, string[] memory _attributes) public {
        Identity memory newIdentity;
        newIdentity.name = _name;
        for (uint i = 0; i < _attributes.length; i++) {
            newIdentity.attributes[_attributes[i]] = '';
        }

        uint256 identityId = identities[msg.sender].length;
        identities[msg.sender].push(newIdentity);

        emit IdentityCreated(msg.sender, identityId, _name, _attributes);
    }

    function updateIdentity(uint256 _identityId, string memory _name, string[] memory _attributes) public {
        Identity storage identity = identities[msg.sender][_identityId];
        identity.name = _name;
        for (uint i = 0; i < _attributes.length; i++) {
            identity.attributes[_attributes[i]] = '';
        }
    }

    function deleteIdentity(uint256 _identityId) public {
        delete identities[msg.sender][_identityId];
    }

    function getIdentity(uint256 _identityId) public view returns (string memory, string[] memory) {
        Identity storage identity = identities[msg.sender][_identityId];
        string[] memory attributes = new string[](0);
        for (uint i = 0; i < 10; i++) {
            if (bytes(identity.attributes[string(abi.encodePacked('attribute', i))]).length > 0) {
                attributes.push(string(identity.attributes[string(abi.encodePacked('attribute', i))]));
            }
        }

        return (identity.name, attributes);
    }
}
