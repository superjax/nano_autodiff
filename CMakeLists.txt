cmake_minimum_required(VERSION 3.8)
project(nano_mhe)

set(CMAKE_CXX_STANDARD 17)

enable_testing()
find_package(Eigen3 REQUIRED)

find_package(GTest REQUIRED)

add_subdirectory(lib/geometry)

include_directories(include
    lib/geometry/include
    ${EIGEN3_INCLUDE_DIRS}
    ${GTEST_INCLUDE_DIRS})

add_executable(nano_ad_test
    src/test_autodiff.cpp
    src/test_autodiff_dynamics.cpp)
target_link_libraries(nano_ad_test
    gtest
    gtest_main
    ${GTEST_LIBRARIES}
    pthread)
#add_custom_command(
#     TARGET nano_mhe_test
#     COMMENT "Run tests"
#     POST_BUILD
#     COMMAND nano_ad_test)
