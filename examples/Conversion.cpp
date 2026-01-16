#include <numbers>

#include <Igor/Logging.hpp>

#include "Container.hpp"

// = Setup =========================================================================================
using Float            = double;
constexpr Index NY     = 100;
constexpr Index NX     = 10 * NY;
constexpr Index NSTEPS = 1000;

constexpr Float X_MIN  = 0.0;
constexpr Float X_MAX  = 10.0;
constexpr Float Y_MIN  = 0.0;
constexpr Float Y_MAX  = (X_MAX - X_MIN) / static_cast<Float>(NX) * static_cast<Float>(NY) + Y_MIN;
constexpr Float T_END  = 10.0;
// = Setup =========================================================================================

// =================================================================================================
struct PhysicalParameters {
  Float dx;
  Float dt;
  Float nu;
  Float rho;
  Float u;
};

struct LatticeParameters {
  static constexpr Float dx = 1.0;
  static constexpr Float dt = 1.0;
  Float tau;
  static constexpr Float rho = 1.0;
  Float u;
  static constexpr Float cs = std::numbers::inv_sqrt3;
};

constexpr auto physical2lattice(const PhysicalParameters& p) -> LatticeParameters {
  const auto Ct   = p.dt;
  const auto Cl   = p.dx;
  const auto Cu   = Cl / Ct;
  const auto Cnu  = Cl * Cu;

  const auto nu_l = Cnu * p.nu;
  return LatticeParameters{
      .tau = 3.0 * nu_l + 0.5,
      .u   = Cu * p.u,
  };
}

auto main() -> int {
  constexpr PhysicalParameters p{
      .dx  = (X_MAX - X_MIN) / static_cast<Float>(NX - 1),
      .dt  = T_END / static_cast<Float>(NSTEPS),
      .nu  = 0.1 / 0.9,
      .rho = 0.9,
      .u   = 1.0,
  };

  constexpr LatticeParameters l = physical2lattice(p);

  Igor::Info("Simulation setup:");
  Igor::Info("  NX     = {}", NX);
  Igor::Info("  NY     = {}", NY);
  Igor::Info("  NSTEPS = {}", NSTEPS);
  Igor::Info("  X_MIN  = {:4.1f}", X_MIN);
  Igor::Info("  X_MAX  = {:4.1f}", X_MAX);
  Igor::Info("  Y_MIN  = {:4.1f}", Y_MIN);
  Igor::Info("  Y_MAX  = {:4.1f}", Y_MAX);
  Igor::Info("  T_END  = {:4.1f}", T_END);
  std::cout << '\n';

  Igor::Info("Physical parameters:");
  Igor::Info("  dx  = {:.6e}", p.dx);
  Igor::Info("  dt  = {:.6e}", p.dt);
  Igor::Info("  nu  = {:.6e}", p.nu);
  Igor::Info("  rho = {:.6e}", p.rho);
  Igor::Info("  u   = {:.6e}", p.u);
  std::cout << '\n';

  Igor::Info("Lattice parameters:");
  Igor::Info("  dx  = {:.6e}", l.dx);
  Igor::Info("  dt  = {:.6e}", l.dt);
  Igor::Info("  tau = {:.6e}", l.tau);
  Igor::Info("  rho = {:.6e}", l.rho);
  Igor::Info("  u   = {:.6e}", l.u);
  Igor::Info("  cs  = {:.6e}", l.cs);
}
