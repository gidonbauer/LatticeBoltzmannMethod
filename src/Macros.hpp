#ifndef LATTICE_BOLTZMANN_METHOD_MACROS_HPP_
#define LATTICE_BOLTZMANN_METHOD_MACROS_HPP_

#if defined(__GNUC__)
#define LBM_COMP_GNUC 1
#else
#define LBM_COMP_GNUC 0
#endif

#if defined(__clang__)
#define LBM_COMP_CLANG 1
#else
#define LBM_COMP_CLANG 0
#endif

#if defined(__llvm__)
#define LBM_COMP_LLVM 1
#else
#define LBM_COMP_LLVM 0
#endif

#if defined(__INTEL_COMPILER)
#define LBM_COMP_INTEL 1
#else
#define LBM_COMP_INTEL 0
#endif

#if LBM_COMP_GNUC
#define LBM_ALWAYS_INLINE __attribute__((flatten)) inline __attribute__((always_inline))
#else
#define LBM_ALWAYS_INLINE inline
#endif

#endif  // LATTICE_BOLTZMANN_METHOD_MACROS_HPP_
