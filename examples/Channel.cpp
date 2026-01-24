#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#include "Container.hpp"
#include "Conversion.hpp"
#include "IO.hpp"
#include "Lattice.hpp"
#include "Monitor.hpp"
#include "Utility.hpp"

// = Setup =========================================================================================
using Float                 = double;
constexpr Index NY          = 100;
constexpr Index NX          = 10 * (NY - 1) + 1;

constexpr Index NSTEPS      = 10000;
constexpr Index WRITE_STEPS = 100;

constexpr Float X_MIN       = 0.0;  // Physical domain size
constexpr Float X_MAX       = 10.0;
constexpr Float Y_MIN       = 0.0;
constexpr Float Y_MAX       = 1.0;
constexpr Float T_END       = 10.0;        // Physical t_end
constexpr Float U_IN        = 1.0;         // Physical wall velocity
constexpr Float RHO         = 0.9;         // Physical density
constexpr Float NU          = 1e-3 / RHO;  // Physical kinematic viscosity
// = Setup =========================================================================================

// =================================================================================================
template <size_t NUM_VEL>
constexpr void
min_max_f(const Field2D<std::array<Float, NUM_VEL>, NX, NY>& f, Float& min, Float& max) {
  static_assert(NX > 0 && NY > 0 && NUM_VEL > 0UZ);

  min = f(0, 0)[0];
  max = f(0, 0)[0];
  for_each_i(f, [&](Index i, Index j) {
    for (auto fi : f(i, j)) {
      if (fi < min) { min = fi; }
      if (fi > max) { max = fi; }
    }
  });
}

// =================================================================================================
auto main() -> int {
  constexpr PhysicalParameters physical_params{
      .dx  = (X_MAX - X_MIN) / static_cast<Float>(NX - 1),
      .dt  = T_END / static_cast<Float>(NSTEPS),
      .rho = RHO,
      .nu  = NU,
  };
  constexpr Conversion convert(physical_params);
  constexpr auto lattice_params = convert.lattice_params();

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
  Igor::Info("  dx  = {:.6e}", physical_params.dx);
  Igor::Info("  dt  = {:.6e}", physical_params.dt);
  Igor::Info("  nu  = {:.6e}", physical_params.nu);
  Igor::Info("  rho = {:.6e}", physical_params.rho);
  Igor::Info("  u   = {:.6e}", U_IN);
  Igor::Info("  Re  = {:.6e}", (Y_MAX - Y_MIN) * U_IN / physical_params.nu);
  std::cout << '\n';
  Igor::Info("Lattice parameters:");
  Igor::Info("  dx  = {:.6e}", lattice_params.dx);
  Igor::Info("  dt  = {:.6e}", lattice_params.dt);
  Igor::Info("  rho = {:.6e}", lattice_params.rho);
  Igor::Info("  cs  = {:.6e}", lattice_params.cs);
  Igor::Info("  nu  = {:.6e}", lattice_params.nu);
  Igor::Info("  tau = {:.6e}", lattice_params.tau);
  Igor::Info("  u   = {:.6e}", convert.velocity_p2l(U_IN));
  Igor::Info("  Re  = {:.6e}",
             static_cast<Float>(NY - 1) * convert.velocity_p2l(U_IN) / lattice_params.nu);
  std::cout << '\n';

  // = Create output directory =====================================================================
  const auto OUTPUT_DIR = get_output_directory();
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  LatticeD2Q9<Float, NX, NY> lattice(convert.lattice_params());
  Field1D<Float, NX> x_physical{};
  Field1D<Float, NY> y_physical{};
  Field2D<Float, NX, NY> rho_physical{};
  Field2D<Float, NX, NY> U_physical{};
  Field2D<Float, NX, NY> V_physical{};

  for_each_a(x_physical,
             [&](Index i) { x_physical(i) = X_MIN + static_cast<Float>(i) * physical_params.dx; });
  for_each_a(y_physical,
             [&](Index j) { y_physical(j) = Y_MIN + static_cast<Float>(j) * physical_params.dx; });

  // = Observation variables - Lattice =============================================================
  Index step      = 0;
  Float f_min     = 0.0;
  Float f_max     = 0.0;
  Float rho_min_l = 0.0;
  Float rho_max_l = 0.0;
  Float U_min_l   = 0.0;
  Float U_max_l   = 0.0;
  Float V_min_l   = 0.0;
  Float V_max_l   = 0.0;

  // = Observation variables - Physical ============================================================
  Float t         = 0.0;
  Float rho_min_p = 0.0;
  Float rho_max_p = 0.0;
  Float U_min_p   = 0.0;
  Float U_max_p   = 0.0;
  Float V_min_p   = 0.0;
  Float V_max_p   = 0.0;

  // = Output ======================================================================================
  Monitor<Float> monitor_lattice(Igor::detail::format("{}/monitor-lattice.log", OUTPUT_DIR));
  monitor_lattice.add_variable(&step, "time");
  monitor_lattice.add_variable(&f_min, "min(f)");
  monitor_lattice.add_variable(&f_max, "max(f)");
  monitor_lattice.add_variable(&rho_min_l, "min(rho)");
  monitor_lattice.add_variable(&rho_max_l, "max(rho)");
  monitor_lattice.add_variable(&U_min_l, "min(U)");
  monitor_lattice.add_variable(&U_max_l, "max(U)");
  monitor_lattice.add_variable(&V_min_l, "min(V)");
  monitor_lattice.add_variable(&V_max_l, "max(V)");

  Monitor<Float> monitor_physical(Igor::detail::format("{}/monitor-physical.log", OUTPUT_DIR));
  monitor_physical.add_variable(&t, "time");
  monitor_physical.add_variable(&rho_min_p, "min(rho)");
  monitor_physical.add_variable(&rho_max_p, "max(rho)");
  monitor_physical.add_variable(&U_min_p, "min(U)");
  monitor_physical.add_variable(&U_max_p, "max(U)");
  monitor_physical.add_variable(&V_min_p, "min(V)");
  monitor_physical.add_variable(&V_max_p, "max(V)");

  DataWriter<Float, NX, NY, 0> data_writer(OUTPUT_DIR, &x_physical, &y_physical);
  data_writer.add_scalar("density", &rho_physical);
  data_writer.add_vector("velocity", &U_physical, &V_physical);
  // = Output ======================================================================================

  // = Initialize ==================================================================================
  fill(lattice.rho, 1.0);
  fill(lattice.U, convert.velocity_p2l(U_IN));
  fill(lattice.V, 0.0);
  for_each_i<Exec::Parallel>(lattice.f, [&](Index i, Index j) {
    for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
      lattice.f(i, j)[q] = f_eq(lattice, i, j, q);
    }
  });
  convert.density_l2p(lattice.rho, rho_physical);
  convert.velocity_l2p(lattice.U, U_physical);
  convert.velocity_l2p(lattice.V, V_physical);
  // = Initialize ==================================================================================

  min_max_f(lattice.f, f_min, f_max);
  rho_min_l = min(lattice.rho);
  rho_max_l = max(lattice.rho);
  U_min_l   = min(lattice.U);
  U_max_l   = max(lattice.U);
  V_min_l   = min(lattice.V);
  V_max_l   = max(lattice.V);
  monitor_lattice.write();
  rho_min_p = min(rho_physical);
  rho_max_p = max(rho_physical);
  U_min_p   = min(U_physical);
  U_max_p   = max(U_physical);
  V_min_p   = min(V_physical);
  V_max_p   = max(V_physical);
  monitor_physical.write();
  data_writer.write(t);

  Igor::ScopeTimer timer("Solver");
  for (step = 1; step <= NSTEPS; ++step) {
    calc_density_and_velocity(lattice);

    collision(lattice);

    // = Boundary on top =======================================================
    for_each<0, NX>([&](Index i) {
      const Index j = NY - 1;
      for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
#if 1
        // Bounce back
        switch (q) {
          case lattice.N:
          case lattice.NW:
          case lattice.NE:
            {
              const Index ii                = i + static_cast<Index>(lattice.cU[q]);
              const Index jj                = j + static_cast<Index>(lattice.cV[q]);
              const size_t q_inv            = lattice.inverse_direction[q];
              lattice.f_star(ii, jj)[q_inv] = lattice.f_star(i, j)[q];
            }
            break;
          default: break;
        }
#else
        // Bounce back moving wall
        switch (q) {
          case lattice.N:
          case lattice.NW:
          case lattice.NE:
            {
              const Index ii               = i + static_cast<Index>(lattice.cU[q]);
              const Index jj               = j + static_cast<Index>(lattice.cV[q]);
              const size_t q_inv           = lattice.inverse_direction[q];

              const Float moving_wall_corr = 2.0 * lattice.w[q] * lattice.rho(i, j) *
                                             (lattice.cU[q] * convert.velocity_p2l(U_WALL)) /
                                             Igor::sqr(lattice.params.cs);
              lattice.f_star(ii, jj)[q_inv] = lattice.f_star(i, j)[q] - moving_wall_corr;
            }
            break;
          default: break;
        }
#endif
      }
    });

    // = Boundary on bottom ====================================================
    for_each<0, NX>([&](Index i) {
      const Index j = 0;
      for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
        // Bounce back
        switch (q) {
          case lattice.S:
          case lattice.SW:
          case lattice.SE:
            {
              const Index ii                = i + static_cast<Index>(lattice.cU[q]);
              const Index jj                = j + static_cast<Index>(lattice.cV[q]);
              const size_t q_inv            = lattice.inverse_direction[q];
              lattice.f_star(ii, jj)[q_inv] = lattice.f_star(i, j)[q];
            }
            break;
          default: break;
        }
      }
    });

    // = Boundary on left ======================================================
    for_each<0, NY>([&](Index j) {
#if 0
      const Index i = 0;
      for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
        // Bounce back
        switch (q) {
          case lattice.W:
          case lattice.SW:
          case lattice.NW:
            {
              const Index ii                = i + static_cast<Index>(lattice.cU[q]);
              const Index jj                = j + static_cast<Index>(lattice.cV[q]);
              const size_t q_inv            = lattice.inverse_direction[q];
              lattice.f_star(ii, jj)[q_inv] = lattice.f_star(i, j)[q];
            }
            break;
          default: break;
        }
      }
#else
      for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
        // Periodic
        lattice.f_star(-1, j)[q] = lattice.f_star(NX - 1, j)[q];
      }
#endif
    });

    // = Boundary on right =====================================================
    for_each<0, NY>([&](Index j) {
#if 0
      const Index i = NX - 1;
      for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
        // Bounce back
        switch (q) {
          case lattice.E:
          case lattice.SE:
          case lattice.NE:
            {
              const Index ii                = i + static_cast<Index>(lattice.cU[q]);
              const Index jj                = j + static_cast<Index>(lattice.cV[q]);
              const size_t q_inv            = lattice.inverse_direction[q];
              lattice.f_star(ii, jj)[q_inv] = lattice.f_star(i, j)[q];
            }
            break;
          default: break;
        }
      }
#else
      for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
        // Periodic
        lattice.f_star(NX, j)[q] = lattice.f_star(0, j)[q];
      }
#endif
    });

    streaming(lattice);

    convert.density_l2p(lattice.rho, rho_physical);
    convert.velocity_l2p(lattice.U, U_physical);
    convert.velocity_l2p(lattice.V, V_physical);

    t = convert.time_l2p(step);
    min_max_f(lattice.f, f_min, f_max);
    rho_min_l = min(lattice.rho);
    rho_max_l = max(lattice.rho);
    U_min_l   = min(lattice.U);
    U_max_l   = max(lattice.U);
    V_min_l   = min(lattice.V);
    V_max_l   = max(lattice.V);
    monitor_lattice.write();
    rho_min_p = min(rho_physical);
    rho_max_p = max(rho_physical);
    U_min_p   = min(U_physical);
    U_max_p   = max(U_physical);
    V_min_p   = min(V_physical);
    V_max_p   = max(V_physical);
    monitor_physical.write();
    if (should_save(step, WRITE_STEPS, NSTEPS)) { data_writer.write(t); }
  }
  Igor::Info("Simulation complete.");
}
