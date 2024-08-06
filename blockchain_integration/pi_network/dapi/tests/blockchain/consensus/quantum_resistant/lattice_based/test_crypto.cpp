#include "crypto.h"
#include <catch2/catch.hpp>

TEST_CASE("Lattice-based cryptography tests") {
  SECTION("Key generation") {
    LatticeCrypto crypto;
    crypto.generate_keys();
    REQUIRE(crypto.public_key() != "");
    REQUIRE(crypto.private_key() != "");
  }

  SECTION("Encryption and decryption") {
    LatticeCrypto crypto;
    crypto.generate_keys();
    std::string plaintext = "Hello, lattice-based cryptography!";
    std::string ciphertext = crypto.encrypt(plaintext);
    std::string decrypted_text = crypto.decrypt(ciphertext);
    REQUIRE(decrypted_text == plaintext);
  }

  SECTION("Digital signatures") {
    LatticeCrypto crypto;
    crypto.generate_keys();
    std::string message = "This is a test message";
    std::string signature = crypto.sign(message);
    bool verified = crypto.verify(message, signature);
    REQUIRE(verified);
  }
}
