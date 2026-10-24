# Xorcist - it casts out the xruns.
#
# Deliberately self-contained: the only dependency is single-precision
# FFTW.  Nothing here may include a host header, so the library can be
# published, packaged and reviewed on its own - which is also what makes
# it acceptable as an OPTIONAL, cleanly encapsulated feature inside a
# host application.

cmake_minimum_required(VERSION 3.10)
project(xorcist VERSION 0.1.0 LANGUAGES CXX)

option(XORCIST_SHARED "Build Xorcist as a shared library" ON)

# XORCIST_SHARED only has an effect standalone: an embedding host sets
# XORCIST_LIBRARY_TYPE directly (GrandOrgue: STATIC), which always wins.
# Found by actually building this standalone (2026-10-25): with no host
# setting XORCIST_LIBRARY_TYPE, it was empty and add_library() fell back
# to CMake's own BUILD_SHARED_LIBS default (OFF) - silently STATIC,
# contradicting the "standalone default: shared" comment below and the
# POSITION_INDEPENDENT_CODE branch that assumed it, neither of which
# actually controlled the library type.
if(NOT XORCIST_LIBRARY_TYPE)
  if(XORCIST_SHARED)
    set(XORCIST_LIBRARY_TYPE SHARED)
  else()
    set(XORCIST_LIBRARY_TYPE STATIC)
  endif()
endif()

add_library(xorcist ${XORCIST_LIBRARY_TYPE} xorcist.cpp)

# NaN-Netze wieder scharf schalten.  Ein einbettender Host (GrandOrgue)
# baut mit -ffast-math; das schliesst -ffinite-math-only ein, und darunter
# faltet der Uebersetzer jedes isfinite() zur Konstanten `true`.  Die
# Stabilisierung in SolveBin verliert damit genau die Wache, die ihre
# Divergenz abfaengt - hoerbar als endloses, maximal lautes Knacken,
# sobald nach digitaler Stille ein Einsatz kommt (2026-08-09).  Der Code
# prueft inzwischen selbst per Bitmuster (XoFinite); dieser Schalter ist
# der zweite Guertel und kostet nichts Messbares, weil die restlichen
# schnellen Mathematikannahmen erhalten bleiben.
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang|AppleClang|IntelLLVM")
  target_compile_options(xorcist PRIVATE -fno-finite-math-only)
endif()

# The OpenCL host layer (xorcist.cpp, guarded by XO_NO_OPENCL) is dlopen'd
# at runtime, never linked - it is a pure compile-time header dependency on
# CL/cl.h. The win64 MinGW cross-toolchain (CMAKE_SYSTEM_NAME=Windows) has
# no such header in its sysroot, unlike the native Linux build where the
# system OpenCL package provides it. Disabling it here only removes an
# optional device-accelerated solve path; SolveBin's CPU/RLS fallback is
# the one every build already relies on when no OpenCL runtime is present
# at all, so this is a missing-header build gap, not a feature regression.
# Giving Windows users the real GPU path too needs OpenCL headers inside
# the cross sysroot (e.g. vendoring KhronosGroup/OpenCL-Headers the same
# way RtAudio/RtMidi/PortAudio/ZitaConvolver are vendored) - a separate,
# deliberate step, not bundled into this build-unblocking fix.
if(WIN32)
  target_compile_definitions(xorcist PRIVATE XO_NO_OPENCL)
endif()
if(NOT XORCIST_LIBRARY_TYPE)
  # Standalone default: shared, so scripting bindings can dlopen it.
  set_property(TARGET xorcist PROPERTY POSITION_INDEPENDENT_CODE ON)
endif()

target_include_directories(
  xorcist PUBLIC $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}>
                 $<INSTALL_INTERFACE:include>)
set_target_properties(
  xorcist PROPERTIES VERSION ${PROJECT_VERSION} SOVERSION 0
                     PUBLIC_HEADER xorcist.h CXX_STANDARD 17)

# FFTW: an embedding host has usually found it already and passes it in
# (XORCIST_FFTW_*); standalone we look for it ourselves.
if(XORCIST_FFTW_LIBRARIES)
  target_include_directories(xorcist PRIVATE ${XORCIST_FFTW_INCLUDE_DIRS})
  target_link_libraries(xorcist PRIVATE ${XORCIST_FFTW_LIBRARIES})
else()
  find_library(FFTW3F_LIBRARY NAMES fftw3f REQUIRED)
  find_path(FFTW3_INCLUDE_DIR NAMES fftw3.h REQUIRED)
  target_include_directories(xorcist PRIVATE ${FFTW3_INCLUDE_DIR})
  target_link_libraries(xorcist PRIVATE ${FFTW3F_LIBRARY})
endif()

if(CMAKE_PROJECT_NAME STREQUAL PROJECT_NAME)
  include(GNUInstallDirs)
  install(
    TARGETS xorcist
    LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
    PUBLIC_HEADER DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})
endif()
