#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>

#include "geometry/quat.h"

#include "nano_ad/nano_ad.h"
#include "nano_ad/test_common.h"

// Example from "Small Unmanned Aircraft: Theory and Practice" by Beard, McLain
// Section 8.6 Attitude Estimation
struct FixedWingAttitudeEstimation
{
  template <typename Derived1, typename Derived2, typename Derived3>
  bool f(Derived1 &xdot, const Derived2 &x, const Derived3 &u) const
  {
    typedef typename Derived1::Scalar T;
    T phi = x(0);
    T theta = x(1);
    T p = u(0);
    T q = u(1);
    T r = u(2);
    T Va = u(3);

    xdot(0) = p + q * sin(phi) * tan(theta) + r * cos(phi) * tan(theta);
    xdot(1) = q * cos(phi) - r * sin(phi);

    return true;
  }
};

Matrix2d fixedWingAnalyticalJacobianDfDx(const Vector2d &x, const Vector4d &u)
{
  double phi = x(0);
  double theta = x(1);
  double p = u(0);
  double q = u(1);
  double r = u(2);
  double Va = u(3);

  Matrix2d dfdx;
  dfdx(0, 0) = q * cos(phi) * tan(theta) - r * sin(phi) * tan(theta);
  dfdx(0, 1) = (q * sin(phi) + r * cos(phi)) / (cos(theta) * cos(theta));
  dfdx(1, 0) = -q * sin(phi) - r * cos(phi);
  dfdx(1, 1) = 0;

  return dfdx;
}

Matrix<double, 2, 4> fixedWingAnalyticalJacobianDfDu(const Vector2d &x,
                                                     const Vector4d &u)
{
  double phi = x(0);
  double theta = x(1);
  double p = u(0);
  double q = u(1);
  double r = u(2);
  double Va = u(3);

  Matrix<double, 2, 4> dfdu;
  dfdu(0, 0) = 1.;
  dfdu(0, 1) = sin(phi) * tan(theta);
  dfdu(0, 2) = cos(phi) * tan(theta);
  dfdu(0, 3) = 0.;
  dfdu(1, 0) = 0.;
  dfdu(1, 1) = cos(phi);
  dfdu(1, 2) = -sin(phi);
  dfdu(1, 3) = 0.;

  return dfdu;
}

TEST(Autodiff, FixedWingAttitudeEstimationExample)
{
  CostFunctorAutoDiff<double, FixedWingAttitudeEstimation, 2, 2, 4> dynamics;

  Vector2d x{ 0.1, -0.2 };
  Vector4d u{ 0.5, -0.4, 0.03, 20 };
  Vector2d xdot;
  Matrix2d dfdx;
  Matrix<double, 2, 4> dfdu;

  dynamics.Evaluate(xdot, x, u, dfdx, dfdu);

  Matrix2d true_dfdx = fixedWingAnalyticalJacobianDfDx(x, u);
  Matrix<double, 2, 4> true_dfdu = fixedWingAnalyticalJacobianDfDu(x, u);

  ASSERT_MAT_NEAR(dfdx, true_dfdx, 1e-5);
  ASSERT_MAT_NEAR(dfdu, true_dfdu, 1e-5);
}

