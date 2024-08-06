#ifndef CRYPTO_H
#define CRYPTO_H

#include <iostream>
#include <vector>
#include <NTL/ZZ.h>
#include <NTL/mat_ZZ.h>

class Crypto {
public:
  Crypto();
  ~Crypto();

  // Hash functions
  std::vector<NTL::ZZ> sha256(const std::vector<NTL::ZZ>& input);
  std::vector<NTL::ZZ> keccak256(const std::vector<NTL::ZZ>& input);

  // Digital signatures
  std::vector<NTL::ZZ> sign(const std::vector<NTL::ZZ>& message, const NTL::ZZ& privateKey);
  bool verify(const std::vector<NTL::ZZ>& message, const std::vector<NTL::ZZ>& signature, const NTL::ZZ& publicKey);

private:
  NTL::ZZ privateKey_;
  NTL::ZZ publicKey_;
};

#endif  // CRYPTO_H
