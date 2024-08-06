#include "crypto.h"
#include <NTL/RR.h>
#include <NTL/mat_RR.h>
#include <NTL/tools.h>

Crypto::Crypto(int dimension, int modulus) : dimension_(dimension), modulus_(modulus) {}

Crypto::~Crypto() {}

void Crypto::generate_keys() {
  // Generate a random matrix A
  NTL::mat_ZZ A;
  A.SetDims(dimension_, dimension_);
  for (int i = 0; i < dimension_; i++) {
    for (int j = 0; j < dimension_; j++) {
      A[i][j] = NTL::RandomBnd(modulus_);
    }
  }

  // Generate a random matrix S
  NTL::mat_ZZ S;
  S.SetDims(dimension_, dimension_);
  for (int i = 0; i < dimension_; i++) {
    for (int j = 0; j < dimension_; j++) {
      S[i][j] = NTL::RandomBnd(modulus_);
    }
  }

  // Compute the public key B = AS
  public_key_ = A * S;

  // Compute the secret key T = S^(-1)
  secret_key_ = S.inv();

  // Compute the trapdoor T' = T mod q
  trapdoor_ = secret_key_ % modulus_;
}

std::vector<NTL::ZZ> Crypto::encrypt(const std::vector<NTL::ZZ>& message) {
  // Convert the message to a matrix
  NTL::mat_ZZ M;
  M.SetDims(1, dimension_);
  for (int i = 0; i < dimension_; i++) {
    M[0][i] = message[i];
  }

  // Encrypt the message using the public key
  NTL::mat_ZZ C = public_key_ * M;

  // Return the ciphertext
  std::vector<NTL::ZZ> ciphertext;
  for (int i = 0; i < dimension_; i++) {
    ciphertext.push_back(C[0][i]);
  }
  return ciphertext;
}

std::vector<NTL::ZZ> Crypto::decrypt(const std::vector<NTL::ZZ>& ciphertext) {
  // Convert the ciphertext to a matrix
  NTL::mat_ZZ C;
  C.SetDims(1, dimension_);
  for (int i = 0; i < dimension_; i++) {
    C[0][i] = ciphertext[i];
  }

  // Decrypt the ciphertext using the secret key
  NTL::mat_ZZ M = secret_key_ * C;

  // Return the plaintext
  std::vector<NTL::ZZ> plaintext;
  for (int i = 0; i < dimension_; i++) {
    plaintext.push_back(M[0][i]);
  }
  return plaintext;
}
