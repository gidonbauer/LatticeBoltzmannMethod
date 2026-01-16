#include <numbers>

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#include "Container.hpp"
#include "IO.hpp"
#include "Monitor.hpp"
#include "Utility.hpp"

// = Setup =========================================================================================
using Float              = double;
constexpr Index NX       = 250;
constexpr Index NY       = 50;
constexpr Index NGHOST   = 1;

constexpr Float Re       = 500.0;                   // Reynolds number
constexpr Float H        = static_cast<Float>(NY);  // Channel height
constexpr Float NU       = 5e-4;                    // Kinematic viscosity
constexpr Float TAU      = 3.0 * NU + 0.5;          // Relaxation time
constexpr Float U        = Re * NU / H;             // Inlet velocity

constexpr Float T_END    = 1e5;
constexpr Float DT_WRITE = 1e3;
// = Setup =========================================================================================

// =================================================================================================
class Lattice {
 public:
  static constexpr Float dt = 1.0;
  static constexpr Float dx = 1.0;
  static constexpr Float dy = 1.0;
  static constexpr Float cs = std::numbers::inv_sqrt3 * dx / dt;  // Speed of sound
  Field1D<Float, NX> x;
  Field1D<Float, NY> y;

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

  Field2D<std::array<Float, NUM_VEL>, NX, NY> f;
  Field2D<std::array<Float, NUM_VEL>, NX, NY, NGHOST> f_star;
  // = D2Q9 ======================================

  Field2D<Float, NX, NY> rho;
  Field2D<Float, NX, NY> U;
  Field2D<Float, NX, NY> V;

  constexpr Lattice() noexcept {
    for_each_a(x, [&](Index i) { x(i) = 0.0 + static_cast<Float>(i) * dx; });
    for_each_a(y, [&](Index j) { y(j) = 0.0 + static_cast<Float>(j) * dy; });
  }
};

// =================================================================================================
template <>
struct std::formatter<Lattice::Dir, char> {
  template <typename ParseContext>
  constexpr auto parse(ParseContext& ctx) const noexcept {
    return ctx.begin();
  }
  template <typename FormatContext>
  constexpr auto format(Lattice::Dir dir, FormatContext& ctx) const noexcept {
    switch (dir) {
      case Lattice::Dir::C:  return std::format_to(ctx.out(), "C");
      case Lattice::Dir::E:  return std::format_to(ctx.out(), "E");
      case Lattice::Dir::W:  return std::format_to(ctx.out(), "W");
      case Lattice::Dir::N:  return std::format_to(ctx.out(), "N");
      case Lattice::Dir::S:  return std::format_to(ctx.out(), "S");
      case Lattice::Dir::NE: return std::format_to(ctx.out(), "NE");
      case Lattice::Dir::SW: return std::format_to(ctx.out(), "SW");
      case Lattice::Dir::NW: return std::format_to(ctx.out(), "NW");
      case Lattice::Dir::SE: return std::format_to(ctx.out(), "SE");
      default:               return std::format_to(ctx.out(), "{}", static_cast<size_t>(dir));
    }
  }
};

// =================================================================================================
constexpr void calc_density_and_velocity(Lattice& lattice) {
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
constexpr auto f_eq(const Lattice& lattice, Index i, Index j, size_t q) {
  IGOR_ASSERT(q < lattice.NUM_VEL,
              "Index q={} is out of bounds. Must be in [{}, {})",
              q,
              0,
              lattice.NUM_VEL);

  const Float u_dot_c = lattice.U(i, j) * lattice.cU[q] + lattice.V(i, j) * lattice.cV[q];
  const Float u_dot_u = Igor::sqr(lattice.U(i, j)) + Igor::sqr(lattice.V(i, j));

  return lattice.w[q] * lattice.rho(i, j) *
         (1.0 + u_dot_c / Igor::sqr(lattice.cs) +
          Igor::sqr(u_dot_c) / (2.0 * Igor::sqr(Igor::sqr(lattice.cs))) -
          u_dot_u / (2.0 * Igor::sqr(lattice.cs)));
}

// =================================================================================================
// template <size_t NUM_VEL>
// constexpr void make_periodic_left_right(Field2D<std::array<Float, NUM_VEL>, NX, NY, 1>& f_star) {
//   // - Left & right --------------------
//   for_each<-1, NY + 1>([&](Index j) {
//     for (size_t q = 0; q < NUM_VEL; ++q) {
//       f_star(-1, j)[q] = f_star(NX - 1, j)[q];
//       f_star(NX, j)[q] = f_star(0, j)[q];
//     }
//   });
//
//   // - Top & bottom --------------------
//   for_each<-1, NX + 1>([&](Index i) {
//     for (size_t q = 0; q < NUM_VEL; ++q) {
//       f_star(i, -1)[q] = f_star(i, NY - 1)[q];
//       f_star(i, NY)[q] = f_star(i, 0)[q];
//     }
//   });
// }

// =================================================================================================
constexpr void collision(Lattice& lattice) {
  // Step 1: Collision
  for_each_i<Exec::Parallel>(lattice.f, [&](Index i, Index j) {
    for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
      lattice.f_star(i, j)[q] =
          lattice.f(i, j)[q] - 1.0 / TAU * (lattice.f(i, j)[q] - f_eq(lattice, i, j, q));
    }
  });
}

// =================================================================================================
constexpr void streaming(Lattice& lattice) {
#if 0
  auto regular_update = [](Lattice& lattice, Index i, Index j, size_t q) {
    const Index ii         = i - static_cast<Index>(lattice.cU[q]);
    const Index jj         = j - static_cast<Index>(lattice.cV[q]);
    lattice.f(i, j)[q] = lattice.f_star(ii, jj)[q];
  };

  // Step 2: Streaming; Assumes lattice units with dt = 1 and dx = dy = 1
  // Assume no-slip wall at top and bottom modelled via bounce back
  for_each_i<Exec::Parallel>(lattice.f, [&](Index i, Index j) {
    if (j == 0) {
      // At bottom
      for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
        switch (q) {
          case lattice.N:
          case lattice.NE:
          case lattice.NW:
            {
              const size_t q_inv     = lattice.inverse_direction[q];
              lattice.f(i, j)[q] = lattice.f_star(i, j)[q_inv];
            }
            break;
          default: regular_update(lattice, i, j, q); break;
        }
      }
    } else if (j == NY - 1) {
      // At top
      for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
        switch (q) {
          case lattice.S:
          case lattice.SE:
          case lattice.SW:
            {
              const size_t q_inv     = lattice.inverse_direction[q];
              lattice.f(i, j)[q] = lattice.f_star(i, j)[q_inv];
            }
            break;
          default: regular_update(lattice, i, j, q); break;
        }
      }
    } else {
      for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
        regular_update(lattice, i, j, q);
      }
    }
  });
#else
  // Step 2: Streaming; Assumes lattice units with dt = 1 and dx = dy = 1
  // Assume no-slip wall at top and bottom modelled via bounce back
  for_each_i<Exec::Parallel>(lattice.f, [&](Index i, Index j) {
    for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
      const Index ii     = i - static_cast<Index>(lattice.cU[q]);
      const Index jj     = j - static_cast<Index>(lattice.cV[q]);
      lattice.f(i, j)[q] = lattice.f_star(ii, jj)[q];
    }
  });
#endif
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
  while (t < T_END) {
    calc_density_and_velocity(lattice);

    collision(lattice);

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

              const Float U_wall           = 0.1;
              const Float moving_wall_corr = 2.0 * lattice.w[q] * lattice.rho(i, j) *
                                             (lattice.cU[q] * U_wall) / Igor::sqr(lattice.cs);
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

    // make_periodic_left_right(lattice.f_star);
    streaming(lattice);

    t += lattice.dt;
    min_max_f(lattice.f, f_min, f_max);
    rho_min = min(lattice.rho);
    U_min   = min(lattice.U);
    V_min   = min(lattice.V);
    rho_max = max(lattice.rho);
    U_max   = max(lattice.U);
    V_max   = max(lattice.V);
    monitor.write();
    if (should_save(t, lattice.dt, DT_WRITE, T_END)) { data_writer.write(t); }
  }
  Igor::Info("Simulation complete.");
}
