#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#include "Container.hpp"
#include "IO.hpp"
#include "Lattice.hpp"
#include "Monitor.hpp"
#include "Utility.hpp"

// = Setup =========================================================================================
using Float                 = double;
constexpr Index NY          = 100;
constexpr Index NX          = 10 * NY;

constexpr Float H           = static_cast<Float>(NY);  // Channel height
constexpr Float NU          = 1e-1;                    // Kinematic viscosity
constexpr Float TAU         = 3.0 * NU + 0.5;          // Relaxation time
constexpr Float U_WALL      = 0.01;                    // Wall velocity
constexpr Float RE          = U_WALL * H / NU;         // Reynolds number

constexpr Index NSTEPS      = 100LL * 100LL;
constexpr Index WRITE_STEPS = 10;
// = Setup =========================================================================================

constexpr auto step2time(Float t_end, Index step) -> Float {
  return t_end / static_cast<Float>(NSTEPS) * static_cast<Float>(step);
}

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
  Igor::Info("H      = {:.6e}", H);
  Igor::Info("NU     = {:.6e}", NU);
  Igor::Info("TAU    = {:.6e}", TAU);
  Igor::Info("U_WALL = {:.6e}", U_WALL);
  Igor::Info("Re     = {:.6e}", RE);

  // = Create output directory =====================================================================
  const auto OUTPUT_DIR = get_output_directory();
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  LatticeD2Q9<Float, NX, NY> lattice{};

  Float t       = 0.0;
  Float f_min   = 0.0;
  Float f_max   = 0.0;
  Float rho_min = 0.0;
  Float rho_max = 0.0;
  Float U_min   = 0.0;
  Float U_max   = 0.0;
  Float V_min   = 0.0;
  Float V_max   = 0.0;

  // = Output ======================================================================================
  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  monitor.add_variable(&t, "time");
  monitor.add_variable(&f_min, "min(f)");
  monitor.add_variable(&f_max, "max(f)");
  monitor.add_variable(&rho_min, "min(rho)");
  monitor.add_variable(&rho_max, "max(rho)");
  monitor.add_variable(&U_min, "min(U)");
  monitor.add_variable(&U_max, "max(U)");
  monitor.add_variable(&V_min, "min(V)");
  monitor.add_variable(&V_max, "max(V)");

  DataWriter<Float, NX, NY, 0> data_writer(OUTPUT_DIR, &lattice.x, &lattice.y);
  data_writer.add_scalar("density", &lattice.rho);
  data_writer.add_vector("velocity", &lattice.U, &lattice.V);
  // = Output ======================================================================================

  // = Initialize ==================================================================================
  fill(lattice.rho, 1.0);
  fill(lattice.U, 0.0);
  fill(lattice.V, 0.0);
  for_each_i<Exec::Parallel>(lattice.f, [&](Index i, Index j) {
    for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
      lattice.f(i, j)[q] = f_eq(lattice, i, j, q);
    }
  });
  // = Initialize ==================================================================================

  min_max_f(lattice.f, f_min, f_max);
  rho_min = min(lattice.rho);
  U_min   = min(lattice.U);
  V_min   = min(lattice.V);
  rho_max = max(lattice.rho);
  U_max   = max(lattice.U);
  V_max   = max(lattice.V);
  monitor.write();
  data_writer.write(t);

  Igor::ScopeTimer timer("Solver");
  for (Index step = 1; step <= NSTEPS; ++step) {
    calc_density_and_velocity(lattice);

    collision(lattice, TAU);

    // = Boundary on top =======================================================
    for_each<0, NX>([&](Index i) {
      const Index j = NY - 1;
      for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
#if 0
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
                                             (lattice.cU[q] * U_WALL) / Igor::sqr(lattice.cs);
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

    t = step2time(10.0, step);
    min_max_f(lattice.f, f_min, f_max);
    rho_min = min(lattice.rho);
    U_min   = min(lattice.U);
    V_min   = min(lattice.V);
    rho_max = max(lattice.rho);
    U_max   = max(lattice.U);
    V_max   = max(lattice.V);
    monitor.write();
    if (should_save(step, WRITE_STEPS, NSTEPS)) { data_writer.write(t); }
  }
  Igor::Info("Simulation complete.");
}
