# pi_voting.py

from solidity import contract, uint256, address, bytes32

@contract
class PiVoting:
    # Mapping of vote IDs to vote data
    votes: {bytes32: Vote} = {}

    # Mapping of user addresses to their vote history
    user_votes: {address: bytes32[]} = {}

    # Event emitted when a new vote is created
    NewVote: event({vote_id: bytes32, title: bytes32, description: bytes32})

    # Event emitted when a user casts a vote
    VoteCast: event({vote_id: bytes32, user: address, choice: uint256})

    def __init__():
        # Initialize the contract
        pass

    def create_vote(title: bytes32, description: bytes32) -> bytes32:
        # Create a new vote with a unique ID
        vote_id = keccak256(title + description)
        votes[vote_id] = Vote(title, description, [])
        emit NewVote(vote_id, title, description)
        return vote_id

    def cast_vote(vote_id: bytes32, choice: uint256) -> bool:
        # Check if the vote exists
        if vote_id not in votes:
            return False

        # Check if the user has already voted
        if vote_id in user_votes[msg.sender]:
            return False

        # Cast the vote
        votes[vote_id].choices.append(choice)
        user_votes[msg.sender].append(vote_id)
        emit VoteCast(vote_id, msg.sender, choice)
        return True

    def get_vote(vote_id: bytes32) -> (bytes32, bytes32, uint256[]):
        # Return the vote data
        return (votes[vote_id].title, votes[vote_id].description, votes[vote_id].choices)

    def get_user_votes(user: address) -> bytes32[]:
        # Return the user's vote history
        return user_votes[user]

class Vote:
    def __init__(title: bytes32, description: bytes32, choices: uint256[]):
        self.title = title
        self.description = description
        self.choices = choices
