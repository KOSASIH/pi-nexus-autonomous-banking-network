from flask import Flask, request, jsonify
from datetime import datetime
import os

app = Flask(__name__)

# In-memory storage for proposals (for demonstration purposes)
proposals = []
proposal_id_counter = 1

class Proposal:
    def __init__(self, title, description, proposer):
        global proposal_id_counter
        self.id = proposal_id_counter
        proposal_id_counter += 1
        self.title = title
        self.description = description
        self.proposer = proposer
        self.created_at = datetime.utcnow()
        self.votes_for = 0
        self.votes_against = 0
        self.status = "Pending"  # Pending, Approved, Rejected

@app.route('/api/proposals', methods=['POST'])
def create_proposal():
    """Create a new community proposal."""
    data = request.json
    title = data.get('title')
    description = data.get('description')
    proposer = data.get('proposer')

    if not title or not description or not proposer:
        return jsonify({"error": "Missing required fields"}), 400

    new_proposal = Proposal(title, description, proposer)
    proposals.append(new_proposal)
    return jsonify({"message": "Proposal created", "proposal_id": new_proposal.id}), 201

@app.route('/api/proposals', methods=['GET'])
def get_proposals():
    """Get a list of all proposals."""
    return jsonify([{
        "id": proposal.id,
        "title": proposal.title,
        "description": proposal.description,
        "proposer": proposal.proposer,
        "created_at": proposal.created_at.isoformat(),
        "votes_for": proposal.votes_for,
        "votes_against": proposal.votes_against,
        "status": proposal.status
    } for proposal in proposals]), 200

@app.route('/api/proposals/<int:proposal_id>/vote', methods=['POST'])
def vote_on_proposal(proposal_id):
    """Vote on a proposal."""
    data = request.json
    vote = data.get('vote')  # True for 'for', False for 'against'

    if vote is None:
        return jsonify({"error": "Vote must be true or false"}), 400

    for proposal in proposals:
        if proposal.id == proposal_id:
            if vote:
                proposal.votes_for += 1
            else:
                proposal.votes_against += 1
            return jsonify({"message": "Vote recorded", "votes_for": proposal.votes_for, "votes_against": proposal.votes_against}), 200

    return jsonify({"error": "Proposal not found"}), 404

@app.route('/api/proposals/<int:proposal_id>/status', methods=['POST'])
def update_proposal_status(proposal_id):
    """Update the status of a proposal."""
    data = request.json
    status = data.get('status')  # Should be 'Approved' or 'Rejected'

    if status not in ['Approved', 'Rejected']:
        return jsonify({"error": "Invalid status"}), 400

    for proposal in proposals:
        if proposal.id == proposal_id:
            proposal.status = status
            return jsonify({"message": "Proposal status updated", "status": proposal.status}), 200

    return jsonify({"error": "Proposal not found"}), 404

if __name__ == '__main__':
    app.run(debug=True)
