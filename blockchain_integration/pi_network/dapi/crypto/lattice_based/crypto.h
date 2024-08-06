#ifndef CRYPTO_H
#define CRYPTO_H

#include <iostream>
#include <vector>
#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>

class Crypto {
public:
  Crypto(int dimension, int modulus);
  ~Crypto();

  // Key generation
  void generate_keys();

  // Encrypt a message
  std::vector<NTL::ZZ> encrypt(const std::vector<NTL::ZZ>& message);

  // Decrypt a ciphertext
  std::vector<NTL::ZZ> decrypt(const std::vector<NTL::ZZ>& ciphertext);

private:
  int dimension_;
  int modulus_;
  NTL::mat_ZZ public_key_;
  NTL::mat_ZZ secret_key_;
  NTL::mat_ZZ trapdoor_;
};

#endif  // CRYPTO_H
