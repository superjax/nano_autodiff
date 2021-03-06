cmake_minimum_required(VERSION 3.8)
project(nano_mhe)

set(CMAKE_CXX_STANDARD 17)

enable_testing()
find_package(Eigen3 REQUIRED)

find_package(GTest REQUIRED)

add_subdirectory(lib/geometry)

add_library(nano_ad INTERFACE)
target_include_directories(nano_ad INTERFACE
    include
    lib/geometry/include
    ${EIGEN3_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS})

add_executable(nano_ad_test
    src/forward_scalar_test.cpp
    src/test_autodiff.cpp
    src/test_autodiff_dynamics.cpp
    src/test_autodiff_local_parameterization.cpp)
target_link_libraries(nano_ad_test
    nano_ad
    gtest
    gtest_main
    ${GTEST_LIBRARIES}
    pthread)
#add_custom_command(
#     TARGET nano_mhe_test
#     COMMENT "Run tests"
#     POST_BUILD
#     COMMAND nano_ad_test)
