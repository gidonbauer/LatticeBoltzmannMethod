#include <numbers>

#include <Igor/Logging.hpp>

#include "Container.hpp"

// = Setup =========================================================================================
using Float            = double;
constexpr Index NY     = 100;
constexpr Index NX     = 10 * NY;
constexpr Index NSTEPS = 10'000;

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

// =================================================================================================
class Conversion {
  Float Ct;
  Float Cl;
  Float Crho;
  Float Cu;
  Float Cnu;

  PhysicalParameters pp;
  LatticeParameters lp;

  static constexpr auto l2p(Index a, Float Ca) -> Float { return l2p(static_cast<Float>(a), Ca); }
  static constexpr auto l2p(Float a, Float Ca) -> Float { return a * Ca; }
  static constexpr auto p2l(Float a, Float Ca) -> Float { return a * Ca; }

 public:
  constexpr Conversion(const PhysicalParameters& p) noexcept
      : Ct(p.dt),
        Cl(p.dx),
        Crho(p.rho),
        Cu(Cl / Ct),
        Cnu(Cl * Cu),
        pp(p),
        lp({
            .tau = 3.0 * visc_p2l(p.nu) + 0.5,
            .u   = velocity_p2l(p.u),
        }) {}

  [[nodiscard]] constexpr auto time_l2p(Index time) const -> Float { return l2p(time, Ct); }
  [[nodiscard]] constexpr auto time_p2l(Float time) const -> Float { return p2l(time, Ct); }

  [[nodiscard]] constexpr auto length_l2p(Index l) const -> Float { return l2p(l, Cl); }
  [[nodiscard]] constexpr auto length_p2l(Float l) const -> Float { return p2l(l, Cl); }

  [[nodiscard]] constexpr auto density_l2p(Float rho) const -> Float { return l2p(rho, Crho); }
  [[nodiscard]] constexpr auto density_p2l(Float rho) const -> Float { return p2l(rho, Crho); }

  [[nodiscard]] constexpr auto velocity_l2p(Float u) const -> Float { return l2p(u, Cu); }
  [[nodiscard]] constexpr auto velocity_p2l(Float u) const -> Float { return p2l(u, Cu); }

  [[nodiscard]] constexpr auto visc_l2p(Float nu) const -> Float { return l2p(nu, Cnu); }
  [[nodiscard]] constexpr auto visc_p2l(Float nu) const -> Float { return p2l(nu, Cnu); }

  [[nodiscard]] constexpr auto physical_params() const -> const PhysicalParameters& { return pp; }
  [[nodiscard]] constexpr auto lattice_params() const -> const LatticeParameters& { return lp; }
};

// =================================================================================================
auto main() -> int {
  constexpr PhysicalParameters p{
      .dx  = (X_MAX - X_MIN) / static_cast<Float>(NX - 1),
      .dt  = T_END / static_cast<Float>(NSTEPS),
      .nu  = 0.1 / 0.9,
      .rho = 0.9,
      .u   = 1.0,
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
