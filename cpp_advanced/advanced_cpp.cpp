#include <iostream>
#include <tuple>
#include <type_traits>

// Template metaprogramming example: compile-time factorial calculation
template <int N> struct Factorial {
  static constexpr int value = N * Factorial<N - 1>::value;
};

template <> struct Factorial<0> {
  static constexprint value = 1;
};

int main() {
  // Modern C++ features: auto, constexpr, and structured bindings
  auto [x, y] = std::make_tuple(10, 20);
  constexpr int result = Factorial<5>::value;
  std::cout << "Factorial of 5: " << result << std::endl;
  std::cout << "x: " << x << ", y: " << y << std::endl;

  return 0;
}
