#include "contract.h"
#include <catch2/catch.hpp>

TEST_CASE("Intelligent contract tests") {
  SECTION("Contract deployment") {
    Contract contract;
    REQUIRE(contract.deploy() == true);
  }

  SECTION("Contract execution") {
    Contract contract;
    contract.deploy();
    std::string input = "Hello, intelligent contract!";
    std::string output = contract.execute(input);
    REQUIRE(output == "Hello, intelligent contract!");
  }

  SECTION("Contract verification") {
    Contract contract;
    contract.deploy();
    std::string input = "Hello, intelligent contract!";
    bool verified = contract.verify(input);
    REQUIRE(verified);
  }
}
