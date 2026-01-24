#include <benchmark/benchmark.h>

#include <Igor/Logging.hpp>
#include <Igor/Math.hpp>
#include <Igor/Timer.hpp>

#include "Container.hpp"
#include "Conversion.hpp"
#include "Lattice.hpp"

// = Setup =========================================================================================
using Float                 = double;
constexpr Index NY          = 100;
constexpr Index NX          = 10 * (NY - 1) + 1;

constexpr Index NSTEPS      = 5000;
constexpr Index BENCH_STEPS = 25;

constexpr Float X_MIN       = 0.0;  // Physical domain size
constexpr Float X_MAX       = 10.0;
// constexpr Float Y_MIN       = 0.0;
// constexpr Float Y_MAX       = 1.0;
constexpr Float T_END  = 10.0;        // Physical t_end
constexpr Float U_WALL = 1.0;         // Physical wall velocity
constexpr Float RHO    = 0.9;         // Physical density
constexpr Float NU     = 1e-1 / RHO;  // Physical kinematic viscosity

constexpr PhysicalParameters physical_params{
    .dx  = (X_MAX - X_MIN) / static_cast<Float>(NX - 1),
    .dt  = T_END / static_cast<Float>(NSTEPS),
    .rho = RHO,
    .nu  = NU,
};
constexpr Conversion convert(physical_params);
// = Setup =========================================================================================

// = Initialize lattice ============================================================================
constexpr void initialize(LatticeD2Q9<Float, NX, NY>& lattice) {
  fill(lattice.rho, 1.0);
  fill(lattice.U, 0.0);
  fill(lattice.V, 0.0);
  for_each_i<Exec::Parallel>(lattice.f, [&](Index i, Index j) {
    for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
      lattice.f(i, j)[q] = f_eq(lattice, i, j, q);
    }
  });
}
// = Initialize lattice ============================================================================

// = Standard for_each =============================================================================
static void bench_for_each_std_algo(benchmark::State& state) noexcept {
  LatticeD2Q9<Float, NX, NY> lattice(convert.lattice_params());
  initialize(lattice);

  for (auto _ : state) {
    for (Index step = 0; step < BENCH_STEPS; ++step) {
      calc_density_and_velocity(lattice);
      collision(lattice);

      // = Boundary on top =======================================================
      for_each<0, NX>([&](Index i) {
        const Index j = NY - 1;
        for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
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
        for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
          // Periodic
          lattice.f_star(-1, j)[q] = lattice.f_star(NX - 1, j)[q];
        }
      });

      // = Boundary on right =====================================================
      for_each<0, NY>([&](Index j) {
        for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
          // Periodic
          lattice.f_star(NX, j)[q] = lattice.f_star(0, j)[q];
        }
      });

      streaming(lattice);
    }
    benchmark::DoNotOptimize(lattice);
  }
}

// = Using regular for loops with OpenMP ===========================================================
template <Index I_MIN, Index I_MAX, ForEachFunc1D FUNC>
LBM_ALWAYS_INLINE void for_each_omp_for(FUNC&& f) noexcept {
#pragma omp parallel for
  for (Index i = I_MIN; i < I_MAX; ++i) {
    f(i);
  }
}

template <typename Float, Index NX, Index NY, ForEachFunc2D FUNC>
LBM_ALWAYS_INLINE void
for_each_i_omp_for([[maybe_unused]] const Field2D<Float, NX, NY, 0, Layout::C>& _,
                   FUNC&& f) noexcept {
#pragma omp parallel for collapse(2)
  for (Index i = 0; i < NX; ++i) {
    for (Index j = 0; j < NY; ++j) {
      f(i, j);
    }
  }
}

template <typename Float, Index NX, Index NY>
constexpr void calc_density_and_velocity_omp_for(LatticeD2Q9<Float, NX, NY>& lattice) {
  for_each_i_omp_for(lattice.rho, [&](Index i, Index j) {
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

template <typename Float, Index NX, Index NY>
constexpr void collision_omp_for(LatticeD2Q9<Float, NX, NY>& lattice) {
  // Step 1: Collision
  for_each_i_omp_for(lattice.f, [&](Index i, Index j) {
    for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
      lattice.f_star(i, j)[q] =
          lattice.f(i, j)[q] -
          1.0 / lattice.params.tau * (lattice.f(i, j)[q] - f_eq(lattice, i, j, q));
    }
  });
}

template <typename Float, Index NX, Index NY>
constexpr void streaming_omp_for(LatticeD2Q9<Float, NX, NY>& lattice) {
  // Step 2: Streaming; Assumes lattice units with dt = 1 and dx = dy = 1
  for_each_i_omp_for(lattice.f, [&](Index i, Index j) {
    for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
      const Index ii     = i - static_cast<Index>(lattice.cU[q]);
      const Index jj     = j - static_cast<Index>(lattice.cV[q]);
      lattice.f(i, j)[q] = lattice.f_star(ii, jj)[q];
    }
  });
}

// = Using regular for loops with OpenMP ===========================================================
static void bench_for_each_omp_for(benchmark::State& state) noexcept {
  LatticeD2Q9<Float, NX, NY> lattice(convert.lattice_params());
  initialize(lattice);

  for (auto _ : state) {
    for (Index step = 0; step < BENCH_STEPS; ++step) {
      calc_density_and_velocity_omp_for(lattice);
      collision_omp_for(lattice);

      // = Boundary on top =======================================================
      for_each_omp_for<0, NX>([&](Index i) {
        const Index j = NY - 1;
        for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
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
        }
      });

      // = Boundary on bottom ====================================================
      for_each_omp_for<0, NX>([&](Index i) {
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
      for_each_omp_for<0, NY>([&](Index j) {
        for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
          // Periodic
          lattice.f_star(-1, j)[q] = lattice.f_star(NX - 1, j)[q];
        }
      });

      // = Boundary on right =====================================================
      for_each_omp_for<0, NY>([&](Index j) {
        for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
          // Periodic
          lattice.f_star(NX, j)[q] = lattice.f_star(0, j)[q];
        }
      });

      streaming_omp_for(lattice);
    }
    benchmark::DoNotOptimize(lattice);
  }
}

// = Manually inlining all the functions ===========================================================
static void bench_for_each_inline(benchmark::State& state) noexcept {
  LatticeD2Q9<Float, NX, NY> lattice(convert.lattice_params());
  initialize(lattice);

  for (auto _ : state) {
    for (Index step = 0; step < BENCH_STEPS; ++step) {
      // calc_density_and_velocity
#pragma omp parallel for collapse(2)
      for (Index i = 0; i < NX; ++i) {
        for (Index j = 0; j < NY; ++j) {
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
        }
      }

      // collision
#pragma omp parallel for collapse(2)
      for (Index i = 0; i < NX; ++i) {
        for (Index j = 0; j < NY; ++j) {
          for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
            lattice.f_star(i, j)[q] =
                lattice.f(i, j)[q] -
                1.0 / lattice.params.tau * (lattice.f(i, j)[q] - f_eq(lattice, i, j, q));
          }
        }
      }

      // = Boundary on top =======================================================
#pragma omp parallel for
      for (Index i = 0; i < NX; ++i) {
        const Index j = NY - 1;
        for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
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
        }
      }

      // = Boundary on bottom ====================================================
#pragma omp parallel for
      for (Index i = 0; i < NX; ++i) {
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
      }

      // = Boundary on left ======================================================
#pragma omp parallel for
      for (Index j = 0; j < NY; ++j) {
        for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
          // Periodic
          lattice.f_star(-1, j)[q] = lattice.f_star(NX - 1, j)[q];
        }
      }

      // = Boundary on right =====================================================
#pragma omp parallel for
      for (Index j = 0; j < NY; ++j) {
        for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
          // Periodic
          lattice.f_star(NX, j)[q] = lattice.f_star(0, j)[q];
        }
      }

      // streaming
#pragma omp parallel for collapse(2)
      for (Index i = 0; i < NX; ++i) {
        for (Index j = 0; j < NY; ++j) {
          for (size_t q = 0; q < lattice.NUM_VEL; ++q) {
            const Index ii     = i - static_cast<Index>(lattice.cU[q]);
            const Index jj     = j - static_cast<Index>(lattice.cV[q]);
            lattice.f(i, j)[q] = lattice.f_star(ii, jj)[q];
          }
        }
      }
    }

    benchmark::DoNotOptimize(lattice);
  }
}

BENCHMARK(bench_for_each_std_algo)->Iterations(10)->Repetitions(5)->Unit(benchmark::kMillisecond);
BENCHMARK(bench_for_each_omp_for)->Iterations(10)->Repetitions(5)->Unit(benchmark::kMillisecond);
BENCHMARK(bench_for_each_inline)->Iterations(10)->Repetitions(5)->Unit(benchmark::kMillisecond);

BENCHMARK_MAIN();
