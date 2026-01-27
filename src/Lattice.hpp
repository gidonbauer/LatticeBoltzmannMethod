#ifndef LATTICE_BOLTZMANN_METHOD_LATTICE_HPP_
#define LATTICE_BOLTZMANN_METHOD_LATTICE_HPP_

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>

#include "Container.hpp"
#include "Conversion.hpp"
#include "ForEach.hpp"

// =================================================================================================
template <typename Float, Index NX, Index NY>
class LatticeD2Q9 {
 public:
  static constexpr Index NGHOST = 1;
  LatticeParameters<Float> params;

  // = D2Q9 ======================================
  enum Dir : size_t { C, E, W, N, S, NE, SW, NW, SE, NUM_VEL };
  static constexpr std::array<Float, NUM_VEL> w{
      4.0 / 9.0,   // 0
      1.0 / 9.0,   // 1
      1.0 / 9.0,   // 2
      1.0 / 9.0,   // 3
      1.0 / 9.0,   // 4
      1.0 / 36.0,  // 5
      1.0 / 36.0,  // 6
      1.0 / 36.0,  // 7
      1.0 / 36.0,  // 8
  };
  static constexpr std::array<Float, NUM_VEL> cU{0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, -1.0, 1.0};
  static constexpr std::array<Float, NUM_VEL> cV{0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, 1.0, -1.0};
  static constexpr std::array<size_t, NUM_VEL> inverse_direction{C, W, E, S, N, SW, NE, SE, NW};

  Field2D<std::array<Float, NUM_VEL>, NX, NY> f;               // Population
  Field2D<std::array<Float, NUM_VEL>, NX, NY, NGHOST> f_star;  // Post collision population
  // = D2Q9 ======================================

  // = Macroscopic quantities ====================
  Field2D<Float, NX, NY> rho;  // Density
  Field2D<Float, NX, NY> U;    // Velocity x
  Field2D<Float, NX, NY> V;    // Velocity y
  // = Macroscopic quantities ====================

  constexpr LatticeD2Q9(LatticeParameters<Float> params) noexcept
      : params(std::move(params)) {}
};

// =================================================================================================
template <typename Float, Index NX, Index NY>
constexpr void calc_density_and_velocity(LatticeD2Q9<Float, NX, NY>& lattice) {
  for_each_i<Exec::Parallel>(lattice.rho, [&](Index i, Index j) {
    lattice.rho(i, j) = 0.0;
    lattice.U(i, j)   = 0.0;
    lattice.V(i, j)   = 0.0;
    for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
      lattice.rho(i, j) += lattice.f(i, j)[q];
      lattice.U(i, j)   += lattice.f(i, j)[q] * lattice.cU[q];
      lattice.V(i, j)   += lattice.f(i, j)[q] * lattice.cV[q];
    }
    if (lattice.rho(i, j) > 1e-8) {
      lattice.U(i, j) /= lattice.rho(i, j);
      lattice.V(i, j) /= lattice.rho(i, j);
    } else {
      lattice.U(i, j) = 0.0;
      lattice.V(i, j) = 0.0;
    }
  });
}

// =================================================================================================
template <typename Float, Index NX, Index NY>
constexpr auto f_eq(const LatticeD2Q9<Float, NX, NY>& lattice, Index i, Index j, size_t q) {
  IGOR_ASSERT(q < lattice.NUM_VEL,
              "Index q={} is out of bounds. Must be in [{}, {})",
              q,
              0,
              static_cast<size_t>(lattice.NUM_VEL));

  const Float u_dot_c = lattice.U(i, j) * lattice.cU[q] + lattice.V(i, j) * lattice.cV[q];
  const Float u_dot_u = Igor::sqr(lattice.U(i, j)) + Igor::sqr(lattice.V(i, j));

  return lattice.w[q] * lattice.rho(i, j) *
         (1.0 + u_dot_c / Igor::sqr(lattice.params.cs) +
          Igor::sqr(u_dot_c) / (2.0 * Igor::sqr(Igor::sqr(lattice.params.cs))) -
          u_dot_u / (2.0 * Igor::sqr(lattice.params.cs)));
}

// =================================================================================================
template <typename Float, Index NX, Index NY>
constexpr void collision(LatticeD2Q9<Float, NX, NY>& lattice) {
  // Step 1: Collision
  for_each_i<Exec::Parallel>(lattice.f, [&](Index i, Index j) {
    for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
      lattice.f_star(i, j)[q] =
          lattice.f(i, j)[q] -
          1.0 / lattice.params.tau * (lattice.f(i, j)[q] - f_eq(lattice, i, j, q));
    }
  });
}

// =================================================================================================
template <typename Float, Index NX, Index NY>
constexpr void streaming(LatticeD2Q9<Float, NX, NY>& lattice) {
  // Step 2: Streaming; Assumes lattice units with dt = 1 and dx = dy = 1
  for_each_i<Exec::Parallel>(lattice.f, [&](Index i, Index j) {
    for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
      const Index ii     = i - static_cast<Index>(lattice.cU[q]);
      const Index jj     = j - static_cast<Index>(lattice.cV[q]);
      lattice.f(i, j)[q] = lattice.f_star(ii, jj)[q];
    }
  });
}

#endif  // LATTICE_BOLTZMANN_METHOD_LATTICE_HPP_
