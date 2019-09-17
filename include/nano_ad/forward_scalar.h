#pragma once

#include <Eigen/Core>
#include <unsupported/Eigen/AutoDiff>

namespace nano_ad {

using namespace Eigen;

template<typename T>
class ForwardAD;

// A shortcut that builds the correct type of expression from the arguments
// This is useful where we are building complicated expressions
template <typename FigureOut>
ForwardAD<FigureOut> autoForwardAD(const typename FigureOut::Scalar &x, const FigureOut& dx)
{
    return ForwardAD<FigureOut>(x,dx);
}

template <typename Type>
using plain = typename internal::remove_all<Type>::type;

template<typename LhsScalar,typename RhsScalar>
using prod_op = internal::scalar_product_op<LhsScalar, RhsScalar>;
template<typename LhsScalar,typename RhsScalar>
using sum_op = internal::scalar_sum_op<LhsScalar, RhsScalar>;
template<typename LhsScalar,typename RhsScalar>
using sub_op = internal::scalar_difference_op<LhsScalar, RhsScalar>;

template<typename _Type>
class ForwardAD
{
public:
    typedef plain<_Type> Dx;
    typedef typename internal::traits<Dx>::Scalar Scalar;
    typedef typename NumTraits<Scalar>::Real Real;

    ForwardAD() {}

    ForwardAD(const Scalar& x, const int dx_idx)
        : x_(x), dx_(Dx::Unit(dx_idx))
    {}

    ForwardAD(const ForwardAD& other) :
        x_(other.x()), dx_(other.dx())
    {}

    ForwardAD(const Scalar& x, const Dx& dx) :
        x_(x), dx_(dx)
    {}

    template<typename T2>
    ForwardAD(const ForwardAD<T2>& other
    , typename internal::enable_if<
            internal::is_same<Scalar, typename internal::traits<plain<T2>>::Scalar>::value
        &&  internal::is_convertible<T2,Dx>::value , void*>::type = 0
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
    const ForwardAD<Dx&> operator+(const Scalar& other) const
    {
        return ForwardAD<Dx&>(x_ + other, dx_);
    }
    friend const ForwardAD<Dx&> operator+(const Scalar& a, const ForwardAD& b)
    {
        return ForwardAD<Dx&>(a + b.x(), b.dx());
    }
    ForwardAD& operator += (const Scalar& other)
    {
        x_ += other;
        return *this;
    }

    const ForwardAD<Dx&> operator-(const Scalar& other) const
    {
        return ForwardAD<Dx&>(x_ - other, dx_);
    }
    friend const ForwardAD<Dx&> operator-(const Scalar& a, const ForwardAD& b)
    {
        return ForwardAD<Dx&>(a - b.x(), b.dx());
    }
    ForwardAD& operator -= (const Scalar& other)
    {
        x_ -= other;
        return *this;
    }

    const ForwardAD<Dx&> operator*(const Scalar& other) const
    {
        return ForwardAD<Dx&>(x_ * other, dx_);
    }
    friend const ForwardAD<Dx&> operator*(const Scalar& a, const ForwardAD& b)
    {
        return ForwardAD<Dx&>(a * b.x(), b.dx());
    }
    ForwardAD& operator *= (const Scalar& other)
    {
        x_ *= other;
        return *this;
    }

    const ForwardAD<Dx&> operator/(const Scalar& other) const
    {
        return ForwardAD<Dx&>(x_ / other, dx_);
    }
    friend const ForwardAD<Dx&> operator/(const Scalar& a, const ForwardAD& b)
    {
        return ForwardAD<Dx&>(a / b.x(), b.dx());
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

    Dx& dx() { return dx_; }
    const Dx& dx() const { return dx_; }

    /// Vector Ops
    // An expression that is the sum of two derivative types (vectors or
    // other sub-expressions)
    template <typename T2>
    using sumExpr = CwiseBinaryOp<sum_op<Scalar, Scalar>,
                           const Dx,
                           const plain<T2>>;
    template<typename T2>
    const ForwardAD<sumExpr<T2>>
    operator+(const ForwardAD<T2>& other) const
    {
        return ForwardAD<sumExpr<T2>>(x_ + other.x(), dx_ + other.dx());
    }


    template<typename Vec2>
    ForwardAD& operator+=(const ForwardAD<Vec2>& other)
    {
      (*this) = (*this) + other;
      return *this;
    }

    // An expression that is the subtraction of two derivative types (vectors or
    // other sub-expressions)
    template <typename T2>
    using subExpr = CwiseBinaryOp<sub_op<Scalar, Scalar>,
                          const Dx,
                          const typename internal::remove_all<T2>::type>;

    template<typename Vec2>
    const ForwardAD<subExpr<Vec2>>
    operator-(const ForwardAD<Vec2>& other) const
    {
        return ForwardAD<subExpr<Vec2>>(x_ - other.x(), dx_ - other.dx());
    }

    template<typename Vec2>
    ForwardAD&  operator-=(const ForwardAD<Vec2>& other) const
    {
        (*this) = (*this) - other;
        return *this;
    }

//    template<typename OtherDerType>
//    using multType = CwiseBinaryOp<internal::scalar_sum_op<Scalar>,
//        const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(Vec,Scalar,product),
//        const EIGEN_EXPR_BINARYOP_SCALAR_RETURN_TYPE(typename internal::remove_all<OtherDerType>::type,Scalar,product)>;

//    template<typename Vec2>
//    multType<Vec2> operator*(const ForwardAD<Vec2>& other) const
//    {
//        return autoForwardAD(x_ * other.x(),
//                             (dx_ * other.x()) + (other.dx() * x_));
//    }

    // Builds the following type of expression
    // (V op S) where V is a vector, S is a scalar and op is the operation
    template <typename V,  typename S, template<typename S1, typename S2> typename op>
    using BinOpExprScalar = CwiseBinaryOp<op<typename internal::traits<V>::Scalar, S>,
                                const V,
                                const typename internal::plain_constant_type<V,S>::type>;



    template<typename Dx2>
    const ForwardAD<CwiseBinaryOp<sum_op<Scalar, Scalar>,
                                  const BinOpExprScalar<Dx, Scalar, prod_op>,
                                  const BinOpExprScalar<plain<Dx2>, Scalar, prod_op>>>
    operator*(const ForwardAD<Dx2>& other) const
    {
      return autoForwardAD(x_ * other.x(), (dx_ * other.x()) + (other.dx() * x_));
    }

    template <typename Dx2>
    const ForwardAD<BinOpExprScalar<
            CwiseBinaryOp<sub_op<Scalar, Scalar>,
              const BinOpExprScalar<Dx,Scalar,prod_op>,
              const BinOpExprScalar<plain<Dx2>,Scalar,prod_op> >,Scalar,prod_op> >
    operator/(const ForwardAD<Dx2>& other) const
    {
        return autoForwardAD(x_ / other.x(),
                             ((dx_ * other.x()) - (other.dx() * x_)) *
                             (Scalar(1)/(other.x()*other.x())));
    }









protected:
    Scalar x_;
    Dx dx_;

};


}
