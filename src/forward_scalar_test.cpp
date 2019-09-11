#include "nano_ad/forward_scalar.h"

#include <gtest/gtest.h>

using namespace nano_ad;

TEST(ForwardScalar, Compile)
{
    ForwardAD<Vector3d> x;
    EXPECT_FLOAT_EQ(x.x(), 0.0);
    EXPECT_FLOAT_EQ(x.dx()(0), 0.0);
    EXPECT_FLOAT_EQ(x.dx()(1), 0.0);
    EXPECT_FLOAT_EQ(x.dx()(2), 0.0);
}

TEST(ForwardScalar, Initialize)
{
    ForwardAD<Vector3d> x(3.0, 2);

    EXPECT_FLOAT_EQ(x.x(), 3.0);
    EXPECT_FLOAT_EQ(x.dx()(0), 0.0);
    EXPECT_FLOAT_EQ(x.dx()(1), 0.0);
    EXPECT_FLOAT_EQ(x.dx()(2), 1.0);
}

TEST(ForwardScalar, Print)
{
    ForwardAD<Vector3d> x(3.0, 2);

    testing::internal::CaptureStdout();
    std::cout << x;
    std::string output = testing::internal::GetCapturedStdout();
    EXPECT_STREQ(output.c_str(), "3");
}

TEST(ForwardScalar, CopyConstructor)
{
    ForwardAD<Vector3d> x(3.0, 2);
    ForwardAD<Vector3d> y(x);

    EXPECT_FLOAT_EQ(y.x(), 3.0);
    EXPECT_FLOAT_EQ(y.dx()(0), 0.0);
    EXPECT_FLOAT_EQ(y.dx()(1), 0.0);
    EXPECT_FLOAT_EQ(y.dx()(2), 1.0);
}

TEST(ForwardScalar, ScalarAdd)
{
    ForwardAD<Vector3d> x(3.0, 2);
    double y = 5.0;
    ForwardAD<Vector3d> z1 = x+y;
    ForwardAD<Vector3d> z2 = y+x;
    x += y;

    EXPECT_FLOAT_EQ(z1.x(), 8.0);
    EXPECT_FLOAT_EQ(z1.dx()(0), 0.0);
    EXPECT_FLOAT_EQ(z1.dx()(1), 0.0);
    EXPECT_FLOAT_EQ(z1.dx()(2), 1.0);
    EXPECT_FLOAT_EQ(z2.x(), 8.0);
    EXPECT_FLOAT_EQ(z2.dx()(0), 0.0);
    EXPECT_FLOAT_EQ(z2.dx()(1), 0.0);
    EXPECT_FLOAT_EQ(z2.dx()(2), 1.0);
    EXPECT_FLOAT_EQ(x.x(), 8.0);
    EXPECT_FLOAT_EQ(x.dx()(0), 0.0);
    EXPECT_FLOAT_EQ(x.dx()(1), 0.0);
    EXPECT_FLOAT_EQ(x.dx()(2), 1.0);
}

TEST(ForwardScalar, ScalarSubtract)
{
    ForwardAD<Vector3d> x(3.0, 2);
    double y = 5.0;
    ForwardAD<Vector3d> z1 = x-y;
    ForwardAD<Vector3d> z2 = y-x;
    x -= y;

    EXPECT_FLOAT_EQ(z1.x(), -2.0);
    EXPECT_FLOAT_EQ(z1.dx()(0), 0.0);
    EXPECT_FLOAT_EQ(z1.dx()(1), 0.0);
    EXPECT_FLOAT_EQ(z1.dx()(2), 1.0);
    EXPECT_FLOAT_EQ(z2.x(), 2.0);
    EXPECT_FLOAT_EQ(z2.dx()(0), 0.0);
    EXPECT_FLOAT_EQ(z2.dx()(1), 0.0);
    EXPECT_FLOAT_EQ(z2.dx()(2), 1.0);
    EXPECT_FLOAT_EQ(x.x(), -2.0);
    EXPECT_FLOAT_EQ(x.dx()(0), 0.0);
    EXPECT_FLOAT_EQ(x.dx()(1), 0.0);
    EXPECT_FLOAT_EQ(x.dx()(2), 1.0);
}

TEST(ForwardScalar, ScalarMultiply)
{
    ForwardAD<Vector3d> x(3.0, 2);
    double y = 5.0;
    ForwardAD<Vector3d> z1 = x*y;
    ForwardAD<Vector3d> z2 = y*x;
    x *= y;

    EXPECT_FLOAT_EQ(z1.x(), 15.0);
    EXPECT_FLOAT_EQ(z1.dx()(0), 0.0);
    EXPECT_FLOAT_EQ(z1.dx()(1), 0.0);
    EXPECT_FLOAT_EQ(z1.dx()(2), 1.0);
    EXPECT_FLOAT_EQ(z2.x(), 15.0);
    EXPECT_FLOAT_EQ(z2.dx()(0), 0.0);
    EXPECT_FLOAT_EQ(z2.dx()(1), 0.0);
    EXPECT_FLOAT_EQ(z2.dx()(2), 1.0);
    EXPECT_FLOAT_EQ(x.x(), 15.0);
    EXPECT_FLOAT_EQ(x.dx()(0), 0.0);
    EXPECT_FLOAT_EQ(x.dx()(1), 0.0);
    EXPECT_FLOAT_EQ(x.dx()(2), 1.0);
}

TEST(ForwardScalar, ScalarDivide)
{
    ForwardAD<Vector3d> x(3.0, 2);
    double y = 5.0;
    ForwardAD<Vector3d> z1 = x/y;
    ForwardAD<Vector3d> z2 = y/x;
    x /= y;

    EXPECT_FLOAT_EQ(z1.x(), 3.0/5.0);
    EXPECT_FLOAT_EQ(z1.dx()(0), 0.0);
    EXPECT_FLOAT_EQ(z1.dx()(1), 0.0);
    EXPECT_FLOAT_EQ(z1.dx()(2), 1.0);
    EXPECT_FLOAT_EQ(z2.x(), 5.0/3.0);
    EXPECT_FLOAT_EQ(z2.dx()(0), 0.0);
    EXPECT_FLOAT_EQ(z2.dx()(1), 0.0);
    EXPECT_FLOAT_EQ(z2.dx()(2), 1.0);
    EXPECT_FLOAT_EQ(x.x(), 3.0/5.0);
    EXPECT_FLOAT_EQ(x.dx()(0), 0.0);
    EXPECT_FLOAT_EQ(x.dx()(1), 0.0);
    EXPECT_FLOAT_EQ(x.dx()(2), 1.0);
}

TEST(ForwardScalar, VectorAdd)
{
    ForwardAD<Vector3d> x1(3.0, 1);
    ForwardAD<Vector3d> x2(8.0, 1);

    ForwardAD<Vector3d> y = x1 + x2;

    EXPECT_FLOAT_EQ(y.x(), 11.0);
    EXPECT_FLOAT_EQ(y.dx()(0), 0);
    EXPECT_FLOAT_EQ(y.dx()(1), 2);
    EXPECT_FLOAT_EQ(y.dx()(2), 0);
}

TEST(ForwardScalar, VectorSubtract)
{
    ForwardAD<Vector3d> x1(3.0, 1);
    ForwardAD<Vector3d> x2(8.0, 1);

    ForwardAD<Vector3d> y = x1 - x2;

    EXPECT_FLOAT_EQ(y.x(), -5);
    EXPECT_FLOAT_EQ(y.dx()(0), 0);
    EXPECT_FLOAT_EQ(y.dx()(1), 0);
    EXPECT_FLOAT_EQ(y.dx()(2), 0);
}

TEST (ForwardScalar, VectorTimes)
{
    ForwardAD<Vector3d> x1(3.0, 1);
    ForwardAD<Vector3d> x2(8.0, 2);
    ForwardAD<Vector3d> x3(8.0, 2);

    ForwardAD<Vector3d> y = x1*x2;



}




