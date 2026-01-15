#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#include "Container.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Utility.hpp"

// = Setup =========================================================================================
using Float            = double;
constexpr Index NX     = 250;
constexpr Index NY     = 50;
constexpr Index NGHOST = 1;

constexpr Float Re     = 500.0;                   // Reynolds number
constexpr Float H      = static_cast<Float>(NY);  // Channel height
constexpr Float NU     = 5e-4;                    // Kinematic viscosity
constexpr Float TAU    = 9.0 * NU + 0.5;          // Relaxation time
constexpr Float U      = Re * NU / H;             // Inlet velocity

constexpr Float T_END  = 100.0;
// = Setup =========================================================================================

// =================================================================================================
struct D2Q9 {
  static constexpr size_t NUM_VEL = 9;

  enum : size_t { C, E, W, N, S, NE, SW, NW, SE };
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
  static constexpr std::array<Float, NUM_VEL> cU{0.0, 1.0, -1.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0};
  static constexpr std::array<Float, NUM_VEL> cV{0.0, 0.0, 0.0, 1.0, -1.0, 1.0, -1.0, -1.0, 1.0};
  static constexpr std::array<size_t, NUM_VEL> inverse_direction{C, W, E, S, N, SW, NE, SE, NW};

  Field2D<std::array<Float, NUM_VEL>, NX, NY> f;
  Field2D<std::array<Float, NUM_VEL>, NX, NY, NGHOST> f_star;
};

// =================================================================================================
class Lattice {
 public:
  static constexpr Float dt = 1.0;
  static constexpr Float dx = 1.0;
  static constexpr Float dy = 1.0;
  Float cs;
  Field1D<Float, NX> x;
  Field1D<Float, NY> y;
  D2Q9 pop{};

  Field2D<Float, NX, NY> rho;
  Field2D<Float, NX, NY> U;
  Field2D<Float, NX, NY> V;

  constexpr Lattice() noexcept
      : cs(1.0 / 3.0 * Igor::sqr(dx) / Igor::sqr(dt)) {
    for_each_a(x, [&](Index i) { x(i) = 0.0 + static_cast<Float>(i) * dx; });
    for_each_a(y, [&](Index j) { y(j) = 0.0 + static_cast<Float>(j) * dy; });
  }
};

// =================================================================================================
constexpr void calc_density_and_velocity(Lattice& lattice) {
  for_each_i<Exec::Parallel>(lattice.rho, [&](Index i, Index j) {
    lattice.rho(i, j) = 0.0;
    lattice.U(i, j)   = 0.0;
    lattice.V(i, j)   = 0.0;
    for (size_t q = 0; q < lattice.pop.NUM_VEL; ++q) {
      lattice.rho(i, j) += lattice.pop.f(i, j)[q];
      lattice.U(i, j)   += lattice.pop.f(i, j)[q] * lattice.pop.cU[q];
      lattice.V(i, j)   += lattice.pop.f(i, j)[q] * lattice.pop.cV[q];
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
constexpr auto f_eq(const Lattice& lattice, Index i, Index j, size_t q) {
  IGOR_ASSERT(q < lattice.pop.NUM_VEL,
              "Index q={} is out of bounds. Must be in [{}, {})",
              q,
              0,
              lattice.pop.NUM_VEL);

  const Float u_dot_ck = lattice.U(i, j) * lattice.pop.cU[q] + lattice.V(i, j) * lattice.pop.cV[q];
  const Float u_dot_u  = Igor::sqr(lattice.U(i, j)) + Igor::sqr(lattice.V(i, j));

  return lattice.pop.w[q] * lattice.rho(i, j) *
         (1.0 + u_dot_ck / Igor::sqr(lattice.cs) +
          Igor::sqr(u_dot_ck) / (2.0 * Igor::sqr(Igor::sqr(lattice.cs))) -
          u_dot_u / (2.0 * Igor::sqr(lattice.cs)));
}

// =================================================================================================
template <size_t NUM_VEL>
constexpr void make_periodic_left_right(Field2D<std::array<Float, NUM_VEL>, NX, NY, 1>& f_star) {
  // - Left & right --------------------
  for_each<-1, NY + 1>([&](Index j) {
    for (size_t q = 0; q < NUM_VEL; ++q) {
      f_star(-1, j)[q] = f_star(NX - 1, j)[q];
      f_star(NX, j)[q] = f_star(0, j)[q];
    }
  });

  // - Top & bottom --------------------
  // for_each<-1, NX + 1>([&](Index i) {
  //   for (size_t q = 0; q < NUM_VEL; ++q) {
  //     f_star(i, -1)[q] = f_star(i, NY - 1)[q];
  //     f_star(i, NY)[q] = f_star(i, 0)[q];
  //   }
  // });
}

// =================================================================================================
constexpr void collision(Lattice& lattice) {
  // Step 1: Collision
  for_each_i<Exec::Parallel>(lattice.pop.f, [&](Index i, Index j) {
    for (size_t q = 0; q < lattice.pop.NUM_VEL; ++q) {
      lattice.pop.f_star(i, j)[q] =
          lattice.pop.f(i, j)[q] - 1.0 / TAU * (lattice.pop.f(i, j)[q] - f_eq(lattice, i, j, q));
    }
  });
}

// =================================================================================================
constexpr void streaming(Lattice& lattice) {
  auto regular_update = [](Lattice& lattice, Index i, Index j, size_t q) {
    const Index ii         = i + static_cast<Index>(lattice.pop.cU[q]);
    const Index jj         = j + static_cast<Index>(lattice.pop.cV[q]);
    lattice.pop.f(i, j)[q] = lattice.pop.f_star(ii, jj)[q];
  };

  // Step 2: Streaming; Assumes lattice units with dt = 1 and dx = dy = 1
  // Assume no-slip wall at top and bottom modelled via bounce back
  for_each_i<Exec::Parallel>(lattice.pop.f, [&](Index i, Index j) {
    if (j == 0) {
      // At bottom
      for (size_t q = 0; q < lattice.pop.NUM_VEL; ++q) {
        switch (q) {
          case lattice.pop.N:
          case lattice.pop.NE:
          case lattice.pop.NW:
            {
              const size_t q_inv     = lattice.pop.inverse_direction[q];
              lattice.pop.f(i, j)[q] = lattice.pop.f_star(i, j)[q_inv];
            }
            break;
          default: regular_update(lattice, i, j, q); break;
        }
      }
    } else if (j == NY - 1) {
      // At top
      for (size_t q = 0; q < lattice.pop.NUM_VEL; ++q) {
        switch (q) {
          case lattice.pop.S:
          case lattice.pop.SE:
          case lattice.pop.SW:
            {
              const size_t q_inv     = lattice.pop.inverse_direction[q];
              lattice.pop.f(i, j)[q] = lattice.pop.f_star(i, j)[q_inv];
            }
            break;
          default: regular_update(lattice, i, j, q); break;
        }
      }
    } else {
      for (size_t q = 0; q < lattice.pop.NUM_VEL; ++q) {
        regular_update(lattice, i, j, q);
      }
    }
  });
}

// =================================================================================================
auto main() -> int {
  Igor::Info("Re  = {:.6e}", Re);
  Igor::Info("H   = {:.6e}", H);
  Igor::Info("NU  = {:.6e}", NU);
  Igor::Info("TAU = {:.6e}", TAU);
  Igor::Info("U   = {:.6e}", U);

  // = Create output directory =====================================================================
  const auto OUTPUT_DIR = get_output_directory();
  if (!init_output_directory(OUTPUT_DIR)) { return 1; }

  Lattice lattice{};

  Float t       = 0.0;
  Float rho_min = 0.0;
  Float rho_max = 0.0;
  Float U_min   = 0.0;
  Float U_max   = 0.0;
  Float V_min   = 0.0;
  Float V_max   = 0.0;

  // = Output ======================================================================================
  Monitor<Float> monitor(Igor::detail::format("{}/monitor.log", OUTPUT_DIR));
  monitor.add_variable(&t, "time");
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
  for_each_i<Exec::Parallel>(lattice.pop.f, [&](Index i, Index j) {
    for (size_t q = 0; q < lattice.pop.NUM_VEL; ++q) {
      lattice.pop.f(i, j)[q] = f_eq(lattice, i, j, q);
    }
  });
  // = Initialize ==================================================================================

  rho_min = min(lattice.rho);
  U_min   = min(lattice.U);
  V_min   = min(lattice.V);
  rho_max = max(lattice.rho);
  U_max   = max(lattice.U);
  V_max   = max(lattice.V);
  monitor.write();
  data_writer.write(t);

  Igor::ScopeTimer timer("Solver");
  while (t < T_END) {
    calc_density_and_velocity(lattice);

    collision(lattice);
    make_periodic_left_right(lattice.pop.f_star);
    streaming(lattice);

    t       += lattice.dt;
    rho_min  = min(lattice.rho);
    U_min    = min(lattice.U);
    V_min    = min(lattice.V);
    rho_max  = max(lattice.rho);
    U_max    = max(lattice.U);
    V_max    = max(lattice.V);
    monitor.write();
    data_writer.write(t);
  }
  Igor::Info("Simulation complete.");
}
