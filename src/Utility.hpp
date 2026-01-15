#ifndef LATTICE_BOLTZMANN_METHOD_UTILITY_HPP_
#define LATTICE_BOLTZMANN_METHOD_UTILITY_HPP_

#include <Igor/Logging.hpp>

#include "Container.hpp"
#include "ForEach.hpp"

// -------------------------------------------------------------------------------------------------
template <bool INCLUDE_GHOST = false,
          typename Float,
          Index N,
          Index NGHOST,
          typename BinaryFunc,
          typename UnaryFunc>
[[nodiscard]] constexpr auto transform_reduce(const Field1D<Float, N, NGHOST>& vec,
                                              Float init,
                                              BinaryFunc&& reduce,
                                              UnaryFunc&& transform) noexcept -> Float {
  Float res = init;
  if (INCLUDE_GHOST) {
    for_each_a(vec,
               [&res,
                &vec,
                reduce    = std::forward<BinaryFunc>(reduce),
                transform = std::forward<UnaryFunc>(transform)](Index i) {
                 res = reduce(res, transform(vec(i)));
               });
  } else {
    for_each_i(vec,
               [&res,
                &vec,
                reduce    = std::forward<BinaryFunc>(reduce),
                transform = std::forward<UnaryFunc>(transform)](Index i) {
                 res = reduce(res, transform(vec(i)));
               });
  }
  return res;
}

template <bool INCLUDE_GHOST = false,
          typename Float,
          Index NX,
          Index NY,
          Index NGHOST,
          Layout LAYOUT,
          typename BinaryFunc,
          typename UnaryFunc>
[[nodiscard]] constexpr auto transform_reduce(const Field2D<Float, NX, NY, NGHOST, LAYOUT>& mat,
                                              Float init,
                                              BinaryFunc&& reduce,
                                              UnaryFunc&& transform) noexcept -> Float {

  Float res = init;
  if (INCLUDE_GHOST) {
    for_each_a(mat,
               [&mat,
                &res,
                reduce    = std::forward<BinaryFunc>(reduce),
                transform = std::forward<UnaryFunc>(transform)](Index i, Index j) {
                 res = reduce(res, transform(mat(i, j)));
               });
  } else {
    for_each_i(mat,
               [&mat,
                &res,
                reduce    = std::forward<BinaryFunc>(reduce),
                transform = std::forward<UnaryFunc>(transform)](Index i, Index j) {
                 res = reduce(res, transform(mat(i, j)));
               });
  }
  return res;
}

// -------------------------------------------------------------------------------------------------
template <bool INCLUDE_GHOST = false, typename CT>
[[nodiscard]] constexpr auto abs_max(const CT& c) noexcept {
  using Float = std::remove_cvref_t<decltype(*c.get_data())>;
  return transform_reduce<INCLUDE_GHOST>(
      c,
      Float{0},
      [](Float a, Float b) { return std::max(a, b); },
      [](Float x) { return std::abs(x); });
}

// -------------------------------------------------------------------------------------------------
template <bool INCLUDE_GHOST = false, typename CT>
[[nodiscard]] constexpr auto max(const CT& c) noexcept {
  using Float = std::remove_cvref_t<decltype(*c.get_data())>;
  static_assert(std::is_floating_point_v<Float>, "Contained must be a floating point type.");
  return transform_reduce<INCLUDE_GHOST>(
      c,
      -std::numeric_limits<Float>::max(),
      [](Float a, Float b) { return std::max(a, b); },
      [](Float x) { return x; });
}

// -------------------------------------------------------------------------------------------------
template <bool INCLUDE_GHOST = false, typename CT>
[[nodiscard]] constexpr auto min(const CT& c) noexcept {
  using Float = std::remove_cvref_t<decltype(*c.get_data())>;
  static_assert(std::is_floating_point_v<Float>, "Contained must be a floating point type.");
  return transform_reduce<INCLUDE_GHOST>(
      c,
      std::numeric_limits<Float>::max(),
      [](Float a, Float b) { return std::min(a, b); },
      [](Float x) { return x; });
}

#endif  // LATTICE_BOLTZMANN_METHOD_UTILITY_HPP_
