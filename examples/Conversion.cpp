#include <Igor/Logging.hpp>

#include "Container.hpp"
#include "Conversion.hpp"

// = Setup =========================================================================================
using Float            = double;

constexpr Float X_MIN  = 0.0;
constexpr Float X_MAX  = 10.0;
constexpr Float Y_MIN  = 0.0;
constexpr Float Y_MAX  = 1.0;
constexpr Float T_END  = 10.0;
constexpr Float U      = 1.0;

constexpr Index NY     = 100;
constexpr Index NX     = static_cast<Index>((X_MAX - X_MIN) / (Y_MAX - Y_MIN)) * (NY - 1) + 1;
constexpr Index NSTEPS = 10'000;

/*
    (Y_MAX - Y_MIN) / (NY - 1) = (X_MAX - X_MIN) / (NX - 1)
<=> NX - 1 = (X_MAX - X_MIN) / (Y_MAX - Y_MIN) * (NY - 1)
<=> NX = (X_MAX - X_MIN) / (Y_MAX - Y_MIN) * (NY - 1) + 1
*/

// = Setup =========================================================================================

// =================================================================================================
auto main() -> int {
  constexpr PhysicalParameters p{
      .dx  = (X_MAX - X_MIN) / static_cast<Float>(NX - 1),
      .dt  = T_END / static_cast<Float>(NSTEPS),
      .rho = 0.9,
      .nu  = 0.1 / 0.9,
  };
  constexpr Conversion conv(p);
  constexpr LatticeParameters l = conv.lattice_params();

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
  Igor::Info("  dy  = {:.6e}", (Y_MAX - Y_MIN) / static_cast<Float>(NY - 1));
  Igor::Info("  dt  = {:.6e}", p.dt);
  Igor::Info("  nu  = {:.6e}", p.nu);
  Igor::Info("  rho = {:.6e}", p.rho);
  Igor::Info("  u   = {:.6e}", U);
  Igor::Info("  Re  = {:.6e}", (Y_MAX - Y_MIN) * U / p.nu);
  std::cout << '\n';

  Igor::Info("Lattice parameters:");
  Igor::Info("  dx  = {:.6e}", l.dx);
  Igor::Info("  dt  = {:.6e}", l.dt);
  Igor::Info("  rho = {:.6e}", l.rho);
  Igor::Info("  cs  = {:.6e}", l.cs);
  Igor::Info("  nu  = {:.6e}", l.nu);
  Igor::Info("  tau = {:.6e}", l.tau);
  Igor::Info("  u   = {:.6e}", conv.velocity_p2l(U));
  Igor::Info("  Re  = {:.6e}", static_cast<Float>(NY - 1) * conv.velocity_p2l(U) / l.nu);
}
