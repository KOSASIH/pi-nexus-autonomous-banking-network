#include <zcash/IncrementalMerkleTree.hpp>
#include <zcash/Note.hpp>

class ZcashModule {
public:
  void deposit(uint256 amount) {
    // Create a new note
    Note note = Note(amount, getSender());

    // Add the note to the incremental Merkle tree
    incrementalMerkleTree.addNote(note);

    // Emit a deposit event
    emit DepositEvent(getSender(), amount);
  }

  void withdraw(uint256 amount) {
    // Create a new note
    Note note = Note(amount, getSender());

    // Remove the note from the incremental Merkle tree
    incrementalMerkleTree.removeNote(note);

    // Emit a withdrawal event
    emit WithdrawalEvent(getSender(), amount);
  }

private:
  IncrementalMerkleTree incrementalMerkleTree;
};
