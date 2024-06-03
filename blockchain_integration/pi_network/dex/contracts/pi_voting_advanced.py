# pi_voting_advanced.py

from solidity import contract, uint256, address, bytes32, string, mapping, struct
from solidity.types import uint, bool, address
from solidity.events import event
from solidity.modifiers import modifier, onlyOwner, onlyAuthorized

@contract
class PiVotingAdvanced:
    # Mapping of vote IDs to vote data
    votes: {bytes32: Vote} = {}

    # Mapping of user addresses to their vote history
    user_votes: {address: bytes32[]} = {}

    # Mapping of vote IDs to their corresponding vote weights
    vote_weights: {bytes32: uint256} = {}

    # Mapping of user addresses to their vote power (based on their PI token balance)
    user_vote_power: {address: uint256} = {}

    # Event emitted when a new vote is created
    NewVote: event({vote_id: bytes32, title: string, description: string, choices: string[]})

    # Event emitted when a user casts a vote
    VoteCast: event({vote_id: bytes32, user: address, choice: uint256, weight: uint256})

    # Event emitted when a vote is closed
    VoteClosed: event({vote_id: bytes32, result: string})

    # Event emitted when a user's vote power is updated
    VotePowerUpdated: event({user: address, vote_power: uint256})

    # Modifier to restrict access to only the contract owner
    @modifier
    def onlyOwner():
        require(msg.sender == owner, "Only the owner can call this function")
        _

    # Modifier to restrict access to only authorized users
    @modifier
    def onlyAuthorized():
        require(authorized_users[msg.sender], "Only authorized users can call this function")
        _

    # Struct to represent a vote
    struct Vote:
        title: string
        description: string
        choices: string[]
        start_time: uint256
        end_time: uint256
        vote_weight: uint256
        results: uint256[]

    # Constructor function
    def __init__():
        # Initialize the contract owner
        owner = msg.sender

        # Initialize the authorized users mapping
        authorized_users: {address: bool} = {}

    # Function to create a new vote
    @onlyOwner
    def create_vote(title: string, description: string, choices: string[], start_time: uint256, end_time: uint256) -> bytes32:
        # Create a new vote with a unique ID
        vote_id = keccak256(title + description)
        votes[vote_id] = Vote(title, description, choices, start_time, end_time, 0, [])
        emit NewVote(vote_id, title, description, choices)
        return vote_id

    # Function to cast a vote
    @onlyAuthorized
    def cast_vote(vote_id: bytes32, choice: uint256) -> bool:
        # Check if the vote exists
        if vote_id not in votes:
            return False

        # Check if the user has already voted
        if vote_id in user_votes[msg.sender]:
            return False

        # Calculate the user's vote weight based on their PI token balance
        user_vote_weight = user_vote_power[msg.sender]

        # Cast the vote
        votes[vote_id].results[choice] += user_vote_weight
        user_votes[msg.sender].append(vote_id)
        emit VoteCast(vote_id, msg.sender, choice, user_vote_weight)
        return True

    # Function to close a vote
    @onlyOwner
    def close_vote(vote_id: bytes32) -> bool:
        # Check if the vote exists
        if vote_id not in votes:
            return False

        # Calculate the winning choice
        winning_choice = 0
        max_votes = 0
        for i in range(len(votes[vote_id].results)):
            if votes[vote_id].results[i] > max_votes:
                max_votes = votes[vote_id].results[i]
                winning_choice = i

        # Emit the vote closed event
        emit VoteClosed(vote_id, votes[vote_id].choices[winning_choice])
        return True

    # Function to update a user's vote power
    @onlyAuthorized
    def update_vote_power(user: address, vote_power: uint256) -> bool:
        # Update the user's vote power
        user_vote_power[user] = vote_power
        emit VotePowerUpdated(user, vote_power)
        return True

    # Function to get a vote's data
    def get_vote(vote_id: bytes32) -> (string, string, string[], uint256, uint256, uint256, uint256[]):
        # Check if the vote exists
        if vote_id not in votes:
            return ("", "", [], 0, 0, 0, [])

        # Return the vote's data
        return (votes[vote_id].title, votes[vote_id].description, votes[vote_id].choices, votes[vote_id].start_time, votes[vote_id].end_time, votes[vote_id].vote_weight, votes[vote_id].results)

    # Function to check if a user has voted in a vote
    def has_voted(user: address, vote_id: bytes32) -> bool:
        # Check if the user has voted in the specified vote
        return vote_id in user_votes[user]
