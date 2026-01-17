#ifndef LATTICE_BOLTZMANN_METHOD_CONVERSION_HPP_
#define LATTICE_BOLTZMANN_METHOD_CONVERSION_HPP_

#include <numbers>

#include "Container.hpp"

// =================================================================================================
template <typename Float>
struct PhysicalParameters {
  Float dx;
  Float dt;
  Float rho;
  Float nu;
};

template <typename Float>
struct LatticeParameters {
  static constexpr Float dx  = 1.0;
  static constexpr Float dt  = 1.0;
  static constexpr Float rho = 1.0;
  static constexpr Float cs  = std::numbers::inv_sqrt3;
  Float nu;
  Float tau;
};

// =================================================================================================
template <typename Float>
class Conversion {
  Float Ct;
  Float Cl;
  Float Crho;
  Float Cu;
  Float Cnu;

  PhysicalParameters<Float> pp;
  LatticeParameters<Float> lp;

  static constexpr auto l2p(Index a, Float Ca) -> Float { return l2p(static_cast<Float>(a), Ca); }
  static constexpr auto l2p(Float a, Float Ca) -> Float { return a * Ca; }
  static constexpr auto p2l(Float a, Float Ca) -> Float { return a / Ca; }

 public:
  constexpr Conversion(const PhysicalParameters<Float>& p) noexcept
      : Ct(p.dt),
        Cl(p.dx),
        Crho(p.rho),
        Cu(Cl / Ct),
        Cnu(Cl * Cu),
        pp(p),
        lp({
            .nu  = visc_p2l(p.nu),
            .tau = 3.0 * visc_p2l(p.nu) + 0.5,
        }) {}

  [[nodiscard]] constexpr auto time_l2p(Index time) const -> Float { return l2p(time, Ct); }
  [[nodiscard]] constexpr auto time_p2l(Float time) const -> Float { return p2l(time, Ct); }

  [[nodiscard]] constexpr auto length_l2p(Index l) const -> Float { return l2p(l, Cl); }
  [[nodiscard]] constexpr auto length_p2l(Float l) const -> Float { return p2l(l, Cl); }

  [[nodiscard]] constexpr auto density_l2p(Float rho) const -> Float { return l2p(rho, Crho); }
  [[nodiscard]] constexpr auto density_p2l(Float rho) const -> Float { return p2l(rho, Crho); }
  template <Index NX, Index NY>
  constexpr void density_l2p(const Field2D<Float, NX, NY>& rho_lattice,
                             Field2D<Float, NX, NY>& rho_physical) const {
    for_each_i(rho_lattice,
               [&](Index i, Index j) { rho_physical(i, j) = density_l2p(rho_lattice(i, j)); });
  }

  [[nodiscard]] constexpr auto velocity_l2p(Float u) const -> Float { return l2p(u, Cu); }
  [[nodiscard]] constexpr auto velocity_p2l(Float u) const -> Float { return p2l(u, Cu); }
  template <Index NX, Index NY>
  constexpr void velocity_l2p(const Field2D<Float, NX, NY>& u_lattice,
                              Field2D<Float, NX, NY>& u_physical) const {
    for_each_i(u_lattice,
               [&](Index i, Index j) { u_physical(i, j) = velocity_l2p(u_lattice(i, j)); });
  }

  [[nodiscard]] constexpr auto visc_l2p(Float nu) const -> Float { return l2p(nu, Cnu); }
  [[nodiscard]] constexpr auto visc_p2l(Float nu) const -> Float { return p2l(nu, Cnu); }

  [[nodiscard]] constexpr auto physical_params() const -> const PhysicalParameters<Float>& {
    return pp;
  }
  [[nodiscard]] constexpr auto lattice_params() const -> const LatticeParameters<Float>& {
    return lp;
  }
};

#endif  // LATTICE_BOLTZMANN_METHOD_CONVERSION_HPP_
