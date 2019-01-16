#include <Eigen/Dense>
#include <gtest/gtest.h>
#include <iostream>

#include "geometry/quat.h"

#include "nano_ad/nano_ad.h"
#include "nano_ad/test_common.h"

template <typename T>
Matrix<T, 3, 3> skew(const Matrix<T, 3, 1>& v)
{
  Matrix<T, 3, 3> skew_mat;
  skew_mat << T(0.0), -v(2), v(1),
              v(2), T(0.0), -v(0),
              -v(1), v(0), T(0.0);
  return skew_mat;
}

//Matrix3d skew(const Vector3d& v)
//{
  //Matrix3d skew_mat;
  //skew_mat << 0.0, -v(2), v(1),
              //v(2), 0.0, -v(0),
              //-v(1), v(0), 0.0;
  //return skew_mat;
//}

// Example from "Improving the Robustness of Visual-Inertial Extended Kalman
// Filtering: Supplemental Material" by Jackson, et. al.
struct UAVDynamics
{
  template <typename Derived1, typename Derived2, typename Derived3>
  bool f(Derived1 &xdot, const Derived2 &x, const Derived3 &u) const
  {
    // x = [position, velocity, quaternion, accel_bias, gyro_bias, drag_term]^T
    // u = [accel, gyro];
    typedef typename Derived1::Scalar T;
    typedef Matrix<T, 3, 1> Vec3;
    typedef Matrix<T, 3, 3> Mat3;

    Vec3 position_b_I_b = x.template block<3, 1>(0, 0);
    Vec3 velocity_b_I_b = x.template block<3, 1>(3, 0);
    quat::Quat<T> quat_I_b(x.template block<4, 1>(6, 0));
    Vec3 accel_bias = x.template block<3, 1>(10, 0);
    Vec3 gyro_bias = x.template block<3, 1>(13, 0);
    T drag_term = x(16);

    Vec3 u_accel = u.template block<3, 1>(0, 0);
    Vec3 u_gyro = u.template block<3, 1>(3, 0);

    Vec3 e3;
    e3 << T(0.), T(0.), T(1.);
    Vec3 grav;
    grav << T(0.), T(0.), T(9.8);
    Mat3 M;
    M <<  T(1.), T(0.), T(0.),
          T(0.), T(1.), T(0.),
          T(0.), T(0.), T(0.);

    xdot.setZero();
    xdot.template block<3, 1>(0, 0) = quat_I_b.R().transpose() * velocity_b_I_b;
    xdot.template block<3, 1>(3, 0) =
        e3 * e3.transpose() * (u_accel - accel_bias) + quat_I_b.R() * grav -
        drag_term * M * velocity_b_I_b -
        skew<T>(u_gyro - gyro_bias) * velocity_b_I_b;
    xdot.template block<3, 1>(6, 0) = u_gyro - gyro_bias;

    return true;
  }
};


Matrix<double, 16, 16> UAVAnalyticalJacobianDfDx(const Matrix<double, 17, 1> &x,
                                                const Matrix<double, 6, 1> &u)
{
  Vector3d pos = x.block<3, 1>(0, 0);
  Vector3d vel = x.block<3, 1>(3, 0);
  quat::Quatd quat_I_b(x.block<4, 1>(6, 0));
  Vector3d beta_a = x.block<3, 1>(10, 0);
  Vector3d beta_g = x.block<3, 1>(13, 0);
  double drag_term = x(16);

  Vector3d u_accel = u.block<3, 1>(0, 0);
  Vector3d u_gyro = u.block<3, 1>(3, 0);

  Vector3d e3;
  e3 << 0., 0., 1.;

  Vector3d grav_vec;
  grav_vec.setZero();
  grav_vec(2) = 9.8;

  Matrix3d M;
  M.setZero();
  M(0, 0) = 1.;
  M(1, 1) = 1.;

  Matrix<double, 16, 16> dfdx;
  dfdx.setZero();
  dfdx.block<3, 3>(0, 3) = quat_I_b.R().transpose();
  dfdx.block<3, 3>(0, 6) = -quat_I_b.R().transpose() * skew<double>(vel);

  dfdx.block<3, 3>(3, 3) = -drag_term * M - skew<double>(u_gyro - beta_g);
  dfdx.block<3, 3>(3, 6) = skew<double>(quat_I_b.R() * grav_vec);
  dfdx.block<3, 3>(3, 9) = -e3 * e3.transpose();
  dfdx.block<3, 3>(3, 12) = -skew<double>(vel);
  dfdx.block<3, 1>(3, 15) = -M*vel;

  dfdx.block<3, 3>(6, 12) = -Matrix3d::Identity();

  return dfdx;
}

Matrix<double, 16, 6> UAVAnalyticalJacobianDfDu(const Matrix<double, 17, 1> &x,
                                                const Matrix<double, 6, 1> &u)
{
  Vector3d pos = x.block<3, 1>(0, 0);
  Vector3d vel = x.block<3, 1>(3, 0);
  quat::Quatd quat_I_b(x.block<4, 1>(6, 0));
  Vector3d beta_a = x.block<3, 1>(10, 0);
  Vector3d beta_g = x.block<3, 1>(13, 0);
  double drag_term = x(16);

  Vector3d u_accel = u.block<3, 1>(0, 0);
  Vector3d u_gyro = u.block<3, 1>(3, 0);

  Vector3d e3;
  e3 << 0., 0., 1.;

  Vector3d grav_vec;
  grav_vec.setZero();
  grav_vec(2) = 9.8;

  Matrix3d M;
  M.setZero();
  M(0, 0) = 1.;
  M(1, 1) = 1.;

  Matrix<double, 16, 6> dfdu;
  dfdu.setZero();
  // TODO why does paper has negative 1 times all of these
  dfdu.block<3, 3>(3, 0) = e3 * e3.transpose();
  dfdu.block<3, 3>(3, 3) = skew<double>(vel);
  dfdu.block<3, 3>(6, 3) = Matrix3d::Identity();

  return dfdu;
}

struct BoxPlusFunctor
{
  template <typename Derived1, typename Derived2, typename Derived3>
  bool f(Derived1 &y, const Derived2 &x, const Derived3 &delta) const
  {
    typedef typename Derived1::Scalar T;
    typedef Matrix<T, 3, 1> Vec3;

    quat::Quat<T> quat_I_b(x.template block<4, 1>(6, 0));
    Vec3 delta_quat(delta.template block<3, 1>(6, 0));
    quat::Quat<T> new_quat = quat_I_b.boxplus(delta_quat);

    y.template block<6, 1>(0, 0) =
        x.template block<6, 1>(0, 0) + delta.template block<6, 1>(0, 0);
    y.template block<4, 1>(6, 0) = new_quat.arr_;
    y.template block<7, 1>(10, 0) =
        x.template block<7, 1>(10, 0) + delta.template block<7, 1>(9, 0);

    return true;
  }
};

class ErrorStateParameterization
{
public:
  //bool Plus(const double* x, const double* delta, double* x_plus_delta) const;
  bool Plus() const;
  bool ComputeJacobian(const Matrix<double, 17, 1> &x,
                       Matrix<double, 17, 16> &jacobian) const
  {
    double quat_w = x(6);
    double quat_x = x(7);
    double quat_y = x(8);
    double quat_z = x(9);

    jacobian.setZero();
    jacobian.setIdentity();
    jacobian(6, 6) = -quat_x;
    jacobian(6, 7) = -quat_y;
    jacobian(6, 8) = -quat_z;

    jacobian(7, 6) = quat_w;
    jacobian(7, 7) = quat_z;
    jacobian(7, 8) = -quat_y;

    jacobian(8, 6) = -quat_z;
    jacobian(8, 7) = quat_w;
    jacobian(8, 8) = quat_x;

    jacobian(9, 6) = quat_y;
    jacobian(9, 7) = -quat_x;
    jacobian(9, 8) = quat_w;
    return true;
  }
  int GlobalSize() const { return 17; }
  int LocalSize() const { return 16; }
};

Matrix<double, 17, 1> randomUAVState()
{
  Matrix<double, 17, 1> x;
  x.setRandom();

  quat::Quatd quat;
  quat.Random();
  x.block<4, 1>(6, 0) = quat.arr_;
  return x;
}

TEST(Autodiff, UAVLocalParameterization)
{
  CostFunctorAutoDiff<double, UAVDynamics, 16, 17, 6> dynamics;

  Matrix<double, 17, 1> x = randomUAVState();
  //std::cout << "random x: " << std::endl << x << std::endl;
  //Vector2d x{ 0.1, -0.2 };
  //Vector4d u{ 0.5, -0.4, 0.03, 20 };
  Matrix<double, 6, 1> u;
  u.setRandom();
  //Vector3d xdot;
  Matrix<double, 16, 1> xdot;
  //Vector2d xdot;
  //Matrix2d dfdx;
  //Matrix<double, 2, 4> dfdu;
  Matrix<double, 16, 17> dfdx_global;
  Matrix<double, 16, 6> dfdu_global;

  dynamics.Evaluate(xdot, x, u, dfdx_global, dfdu_global);

  CostFunctorAutoDiff<double, BoxPlusFunctor, 17, 17, 16> box_plus;
  Matrix<double, 17, 1> y;
  Matrix<double, 16, 1> delta;
  delta.setZero();
  Matrix<double, 17, 17> bp_dfdx;
  Matrix<double, 17, 16> bp_dfdd;
  box_plus.Evaluate(y, x, delta, bp_dfdx, bp_dfdd);
  std::cout << "bp_dfdx: " << std::endl << bp_dfdx << std::endl;
  std::cout << "bp_dfdd: " << std::endl << bp_dfdd << std::endl;

  // http://ceres-solver.org/nnls_modeling.html#localparameterization
  // To use a local parameterization, or error state parameterization
  // local_matrix = global_matrix * jacobian
  Matrix<double, 17, 16> jacobian;
  ErrorStateParameterization parameterization;
  parameterization.ComputeJacobian(x, jacobian);
  std::cout << "local param jacobian: " << std::endl << jacobian << std::endl;

  Matrix<double, 16, 16> dfdx_local;
  //dfdx_local = dfdx_global * jacobian;
  dfdx_local = dfdx_global * bp_dfdd;

  //Matrix<double, 16, 6> dfdu_local;
  //dfdu_local = dfdu_global * bp_dfdd;

  Matrix<double, 16, 16> true_dfdx_local = UAVAnalyticalJacobianDfDx(x, u);
  Matrix<double, 16, 6> true_dfdu_local = UAVAnalyticalJacobianDfDu(x, u);

  std::cout << "dfdx: " << std::endl << dfdx_local << std::endl;
  std::cout << "true dfdx: " << std::endl << true_dfdx_local << std::endl;

  std::cout << "dfdu: " << std::endl << dfdu_global << std::endl;
  std::cout << "true dfdu: " << std::endl << true_dfdu_local << std::endl;
  ASSERT_MAT_NEAR(dfdx_local, true_dfdx_local, 1e-14);
  ASSERT_MAT_NEAR(dfdu_global, true_dfdu_local, 1e-14);
}

