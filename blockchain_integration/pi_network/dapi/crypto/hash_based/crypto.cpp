#include "crypto.h"
#include <NTL/SHA.h>
#include <NTL/Keccak.h>

Crypto::Crypto() {}

Crypto::~Crypto() {}

std::vector<NTL::ZZ> Crypto::sha256(const std::vector<NTL::ZZ>& input) {
  NTL::SHA256 sha256;
  sha256.update(input);
  std::vector<NTL::ZZ> hash = sha256.final();
  return hash;
}

std::vector<NTL::ZZ> Crypto::keccak256(const std::vector<NTL::ZZ>& input) {
  NTL::Keccak keccak;
  keccak.update(input);
  std::vector<NTL::ZZ> hash = keccak.final();
  return hash;
}

std::vector<NTL::ZZ> Crypto::sign(const std::vector<NTL::ZZ>& message, const NTL::ZZ& privateKey) {
  // Implement digital signature algorithm (e.g. ECDSA)
  return std::vector<NTL::ZZ>();
}

bool Crypto::verify(const std::vector<NTL::ZZ>& message, const std::vector<NTL::ZZ>& signature, const NTL::ZZ& publicKey) {
  // Implement digital signature verification algorithm (e.g. ECDSA)
  return true;
}
