#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

namespace nano_ad {

using namespace Eigen;

template<typename T>
class ForwardAD;

template <typename FigureOut>
ForwardAD<FigureOut> autoForwardAD(const typename FigureOut::Scalar &x, const FigureOut& dx)
{
    return ForwardAD<FigureOut>(x,dx);
}

template<typename _Type>
class ForwardAD
{
public:
    typedef typename internal::remove_all<_Type>::type Vec;
    typedef typename internal::traits<Vec>::Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real Real;

    ForwardAD() {}

    ForwardAD(const Scalar& x, const int dx_idx)
        : x_(x), dx_(Vec::Unit(dx_idx))
    {}

    ForwardAD(const ForwardAD& other) :
        x_(other.x()), dx_(other.dx())
    {}

    ForwardAD(const Scalar& x, const Vec& dx) :
        x_(x), dx_(dx)
    {}

    template<typename Vec2>
    ForwardAD(const ForwardAD<Vec2>& other
    , typename internal::enable_if<
            internal::is_same<Scalar, typename internal::traits<typename internal::remove_all<Vec2>::type>::Scalar>::value
        &&  internal::is_convertible<Vec2,Vec>::value , void*>::type = 0
    )
      : x_(other.x()), dx_(other.dx())
    {}

    template<typename Type2>
    ForwardAD& operator=(const ForwardAD<Type2>& other)
    {
        x_ = other.x();
        dx_ = other.dx_;
        return *this;
    }

    ForwardAD& operator=(const ForwardAD& other)
    {
        x_ = other.x();
        dx_ = other.dx();
        return *this;
    }

    ForwardAD& operator=(const Scalar& other)
    {
        x_ = other.x();
        dx_.setZero();
        return *this;
    }


    // Boolean Ops
    bool operator< (const Scalar& other) const  { return x_ <  other; }
    bool operator<=(const Scalar& other) const  { return x_ <= other; }
    bool operator> (const Scalar& other) const  { return x_ >  other; }
    bool operator>=(const Scalar& other) const  { return x_ >= other; }
    bool operator==(const Scalar& other) const  { return x_ == other; }
    bool operator!=(const Scalar& other) const  { return x_ != other; }
    friend bool operator< (const Scalar& a, const ForwardAD& b) { return a <  b.x(); }
    friend bool operator<=(const Scalar& a, const ForwardAD& b) { return a <= b.x(); }
    friend bool operator> (const Scalar& a, const ForwardAD& b) { return a >  b.x(); }
    friend bool operator>=(const Scalar& a, const ForwardAD& b) { return a >= b.x(); }
    friend bool operator==(const Scalar& a, const ForwardAD& b) { return a == b.x(); }
    friend bool operator!=(const Scalar& a, const ForwardAD& b) { return a != b.x(); }
    template<typename T2> inline bool operator< (const ForwardAD<T2>& b) const  { return x_ <  b.x(); }
    template<typename T2> inline bool operator<=(const ForwardAD<T2>& b) const  { return x_ <= b.x(); }
    template<typename T2> inline bool operator> (const ForwardAD<T2>& b) const  { return x_ >  b.x(); }
    template<typename T2> inline bool operator>=(const ForwardAD<T2>& b) const  { return x_ >= b.x(); }
    template<typename T2> inline bool operator==(const ForwardAD<T2>& b) const  { return x_ == b.x(); }
    template<typename T2> inline bool operator!=(const ForwardAD<T2>& b) const  { return x_ != b.x(); }


    // Scalar Ops
    const ForwardAD<Vec&> operator+(const Scalar& other) const
    {
        return ForwardAD<Vec&>(x_ + other, dx_);
    }
    friend const ForwardAD<Vec&> operator+(const Scalar& a, const ForwardAD& b)
    {
        return ForwardAD<Vec&>(a + b.x(), b.dx());
    }
    ForwardAD& operator += (const Scalar& other)
    {
        x_ += other;
        return *this;
    }

    const ForwardAD<Vec&> operator-(const Scalar& other) const
    {
        return ForwardAD<Vec&>(x_ - other, dx_);
    }
    friend const ForwardAD<Vec&> operator-(const Scalar& a, const ForwardAD& b)
    {
        return ForwardAD<Vec&>(a - b.x(), b.dx());
    }
    ForwardAD& operator -= (const Scalar& other)
    {
        x_ -= other;
        return *this;
    }

    const ForwardAD<Vec&> operator*(const Scalar& other) const
    {
        return ForwardAD<Vec&>(x_ * other, dx_);
    }
    friend const ForwardAD<Vec&> operator*(const Scalar& a, const ForwardAD& b)
    {
        return ForwardAD<Vec&>(a * b.x(), b.dx());
    }
    ForwardAD& operator *= (const Scalar& other)
    {
        x_ *= other;
        return *this;
    }

    const ForwardAD<Vec&> operator/(const Scalar& other) const
    {
        return ForwardAD<Vec&>(x_ / other, dx_);
    }
    friend const ForwardAD<Vec&> operator/(const Scalar& a, const ForwardAD& b)
    {
        return ForwardAD<Vec&>(a / b.x(), b.dx());
    }
    ForwardAD& operator /= (const Scalar& other)
    {
        x_ /= other;
        return *this;
    }

    friend std::ostream& operator << (std::ostream &s, const ForwardAD&a)
    {
        return s << a.x();
    }

    Scalar& x() { return x_; }
    const Scalar& x() const { return x_; }

    Vec& dx() { return dx_; }
    const Vec& dx() const { return dx_; }

    /// Vector Ops
    template <typename Vec2>
    using sumResult = CwiseBinaryOp<internal::scalar_sum_op<Scalar>,
                           const Vec,
                           const typename internal::remove_all<Vec2>::type>;
    template<typename Vec2>
    const ForwardAD<sumResult<Vec2>>
    operator+(const ForwardAD<Vec2>& other) const
    {
        return ForwardAD<sumResult<Vec2>>(x_ + other.x(), dx_ + other.dx());
    }


    template<typename Vec2>
    ForwardAD& operator+=(const ForwardAD<Vec2>& other)
    {
      (*this) = (*this) + other;
      return *this;
    }

    template <typename Vec2>
    using diffResult = CwiseBinaryOp<internal::scalar_difference_op<Scalar>,
                          const Vec,
                          const typename internal::remove_all<Vec2>::type>;

    template<typename Vec2>
    const ForwardAD<diffResult<Vec2>>
    operator-(const ForwardAD<Vec2>& other) const
    {
        return ForwardAD<diffResult<Vec2>>(x_ - other.x(), dx_ - other.dx());
    }

    template<typename Vec2>
    ForwardAD&  operator-=(const ForwardAD<Vec2>& other) const
    {
        (*this) = (*this) - other;
        return *this;
    }

    template<typename Vec2>
    auto operator*(const ForwardAD<Vec2>& other) const
    {
        return autoForwardAD(x_ * other.x(),
                             dx_ * other.x() + other.dx() * x_);
    }







protected:
    Scalar x_;
    Vec dx_;

};


}
