/*===-----------------------------------------------------------------------===
 *
 * Copyright (C) [2022] by Cambricon, Inc.
 *
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 *
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 *
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 *
 *===-----------------------------------------------------------------------===
 */

#ifndef __BANG_FP16_H__
#define __BANG_FP16_H__

#if defined(__BANGCC__)
#define __BANG_HOSTDEVICE_FP16_DECL__ static __mlu_host__ __mlu_func__
#define __BANG_HOSTDEVICE__ __mlu_host__ __mlu_func__ inline
#else /* !defined(__BANGCC__) */
#if defined(__GNUC__)
#define __BANG_HOSTDEVICE_FP16_DECL__ static __attribute__((unused))
#else
#define __BANG_HOSTDEVICE_FP16_DECL__ static
#endif /* defined(__GNUC__) */
#define __BANG_HOSTDEVICE__ inline
#endif /* defined(__BANGCC_) */

#if !defined(_MSC_VER) && __cplusplus >= 201103L
#   define __CPP_VERSION_AT_LEAST_11_FP16
#elif _MSC_FULL_VER >= 190024210 && _MSVC_LANG >= 201103L
#   define __CPP_VERSION_AT_LEAST_11_FP16
#endif

/* Set up structure-alignment attribute */
/* Define alignment macro based on compiler type (cannot assume C11 "_Alignas" is available) */
#if __cplusplus >= 201103L
#define __BANG_ALIGN__(n) alignas(n)    /* C++11 kindly gives us a keyword for this */
#else /* !defined(__CPP_VERSION_AT_LEAST_11_FP16)*/
#if defined(__GNUC__)
#define __BANG_ALIGN__(n) __attribute__ ((aligned(n)))
#elif defined(_MSC_VER)
#define __BANG_ALIGN__(n) __declspec(align(n))
#else
#define __BANG_ALIGN__(n)
#endif /* defined(__GNUC__) */
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP16) */

// define __BANG_FP16_CONSTEXPR__ in order to
// use constexpr where possible, with supporting C++ dialects
// undef after use
#if (defined __CPP_VERSION_AT_LEAST_11_FP16)
#define __BANG_FP16_CONSTEXPR__   constexpr
#else
#define __BANG_FP16_CONSTEXPR__
#endif

typedef struct __BANG_ALIGN__(2) {
  unsigned short x;
} __half_raw;

/* Hide GCC member initialization list warnings because of host/device in-function init requirement */
#if defined(__GNUC__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wstrict-aliasing"
#pragma GCC diagnostic ignored "-Weffc++"
#endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) */
#endif /* defined(__GNUC__) */

/* class' : multiple assignment operators specified
   The class has multiple assignment operators of a single type. This warning is informational */
#if defined(_MSC_VER) && _MSC_VER >= 1500
#pragma warning( push )
#pragma warning( disable:4522 )
#endif /* defined(_MSC_VER) && _MSC_VER >= 1500 */

// forward-declaration of bfloat type to be used in converting constructor
struct __bang_bfloat16;

/* Macros to allow half to be used by inline assembly */
#define __HALF_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __HALF_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))

struct __BANG_ALIGN__(2) __half {
protected:
    /**
     * Protected storage variable contains the bits of floating-point data.
     */
    unsigned short __x;

public:
    /**
     * Constructor by default.
     */
#if defined(__CPP_VERSION_AT_LEAST_11_FP16)
    __half() = default;
#else
    __BANG_HOSTDEVICE__ __half();
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP16) */

    /* Convert to/from __half_raw */
    /**
     * Constructor from \p __half_raw.
     */
    __BANG_HOSTDEVICE__ __BANG_FP16_CONSTEXPR__ __half(const __half_raw &hr) : __x(hr.x) { }
    /**
     * Assignment operator from \p __half_raw.
     */
    __BANG_HOSTDEVICE__ __half &operator=(const __half_raw &hr);
    /**
     * Assignment operator from \p __half_raw to \p volatile \p __half.
     */
    __BANG_HOSTDEVICE__ volatile __half &operator=(const __half_raw &hr) volatile;
    /**
     * Assignment operator from \p volatile \p __half_raw to \p volatile \p __half.
     */
    __BANG_HOSTDEVICE__ volatile __half &operator=(const volatile __half_raw &hr) volatile;
    /**
     * Type cast to \p __half_raw operator.
     */
    __BANG_HOSTDEVICE__ operator __half_raw() const;
    /**
     * Type cast to \p __half_raw operator with \p volatile input.
     */
    __BANG_HOSTDEVICE__ operator __half_raw() const volatile;
    /**
     * Construct \p __half from \p __nv_bfloat16 input using default round-to-nearest-even rounding mode.
     * Need to include the header file \p bang_bf16.h
     */
    __BANG_HOSTDEVICE__ __half(const __bang_bfloat16 f); //forward declaration only, implemented in bang_bf16.hpp
    /* Construct from float/double */
    /**
     * Construct \p __half from \p float input using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half(const float f);
    /**
     * Construct \p __half from \p double input using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half(const double f);
    /**
     * Type cast to \p float operator.
     */
    __BANG_HOSTDEVICE__ operator float() const;
    /**
     * Type cast to \p __half assignment operator from \p float input using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half &operator=(const float f);

    /* We omit "cast to double" operator, so as to not be ambiguous about up-cast */
    /**
     * Type cast to \p __half assignment operator from \p double input using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half &operator=(const double f);

    /* Allow automatic construction from types supported natively in hardware */
    /* Note we do avoid constructor init-list because of special host/device compilation rules */

    /**
     * Construct \p __half from \p short integer input using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half(const short val);
    /**
     * Construct \p __half from \p unsigned \p short integer input using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half(const unsigned short val);
    /**
     * Construct \p __half from \p int input using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half(const int val);
    /**
     * Construct \p __half from \p unsigned \p int input using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half(const unsigned int val);
    /**
     * Construct \p __half from \p long input using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half(const long val);
    /**
     * Construct \p __half from \p unsigned \p long input using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half(const unsigned long val);

    /**
     * Construct \p __half from \p long \p long input using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half(const long long val);
    /**
     * Construct \p __half from \p unsigned \p long \p long input using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half(const unsigned long long val);

    /* Allow automatic casts to supported builtin types, matching all that are permitted with float */

    /**
     * Conversion operator to \p signed \p char data type.
     * Using round-toward-zero rounding mode.
     *
     * See __half2char_rz(__half) for further details
     */
    __BANG_HOSTDEVICE__ operator signed char() const;
    /**
     * Conversion operator to \p unsigned \p char data type.
     * Using round-toward-zero rounding mode.
     *
     * See __half2uchar_rz(__half) for further details
     */
    __BANG_HOSTDEVICE__ operator unsigned char() const;
    /**
     * Conversion operator to an implementation defined \p char data type.
     * Using round-toward-zero rounding mode.
     *
     * Detects signedness of the \p char type and proceeds accordingly, see
     * further details in signed and unsigned char operators.
     */
    __BANG_HOSTDEVICE__ operator char() const;
    /**
     * Conversion operator to \p short data type.
     * Using round-toward-zero rounding mode.
     *
     * See __half2short_rz(__half) for further details
     */
    __BANG_HOSTDEVICE__ operator short() const;
    /**
     * Conversion operator to \p unsigned \p short data type.
     * Using round-toward-zero rounding mode.
     *
     * See __half2ushort_rz(__half) for further details
     */
    __BANG_HOSTDEVICE__ operator unsigned short() const;
    /**
     * Conversion operator to \p int data type.
     * Using round-toward-zero rounding mode.
     *
     * See __half2int_rz(__half) for further details
     */
    __BANG_HOSTDEVICE__ operator int() const;
    /**
     * Conversion operator to \p unsigned \p int data type.
     * Using round-toward-zero rounding mode.
     *
     * See __half2uint_rz(__half) for further details
     */
    __BANG_HOSTDEVICE__ operator unsigned int() const;
    /**
     * Conversion operator to \p long data type.
     * Using round-toward-zero rounding mode.
     */
    __BANG_HOSTDEVICE__ operator long() const;
    /**
     * Conversion operator to \p unsigned \p long data type.
     * Using round-toward-zero rounding mode.
     */
    __BANG_HOSTDEVICE__ operator unsigned long() const;
    /**
     * Conversion operator to \p long \p long data type.
     * Using round-toward-zero rounding mode.
     *
     * See __half2ll_rz(__half) for further details
     */
    __BANG_HOSTDEVICE__ operator long long() const;
    /**
     * Conversion operator to \p unsigned \p long \p long data type.
     * Using round-toward-zero rounding mode.
     *
     * See __half2ull_rz(__half) for further details
     */
    __BANG_HOSTDEVICE__ operator unsigned long long() const;
    /**
     * Type cast from \p short assignment operator, using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half &operator=(const short val);
    /**
     * Type cast from \p unsigned \p short assignment operator, using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half &operator=(const unsigned short val);
    /**
     * Type cast from \p int assignment operator, using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half &operator=(const int val);
    /**
     * Type cast from \p unsigned \p int assignment operator, using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half &operator=(const unsigned int val);
    /**
     * Type cast from \p long \p long assignment operator, using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half &operator=(const long long val);
    /**
     * Type cast from \p long assignment operator, using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half &operator=(const long val);
    /**
     * Type cast from \p unsigned \p long assignment operator, using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half &operator=(const unsigned long val);
    /**
     * Type cast from \p unsigned \p long \p long assignment operator, using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half &operator=(const unsigned long long val);
    /**
     * Type cast from \p __bang_bfloat16 assignment operator, using default round-to-nearest-even rounding mode.
     */
    __BANG_HOSTDEVICE__ __half &operator=(const __bang_bfloat16 f);
    /**
     * Conversion operator to \p bool data type.
     * +0 and -0 inputs convert to \p false.
     * Non-zero inputs convert to \p true.
     */
    __BANG_HOSTDEVICE__ __BANG_FP16_CONSTEXPR__ operator bool() const { return (__x & 0x7FFFU) != 0U; }
};

__BANG_HOSTDEVICE_FP16_DECL__ __half __float2half__(const float a);
__BANG_HOSTDEVICE_FP16_DECL__ float __half2float(const __half a);
__BANG_HOSTDEVICE_FP16_DECL__ short int __half2short_rz(const __half h);

__BANG_HOSTDEVICE_FP16_DECL__ long long int __half2ll_rz(const __half h);
__BANG_HOSTDEVICE_FP16_DECL__ unsigned long long int __half2ull_rz(const __half h);
__BANG_HOSTDEVICE_FP16_DECL__ __half __short2half_rn__(const short int i);
__BANG_HOSTDEVICE_FP16_DECL__ __half __ushort2half_rn__(const unsigned short int i);

__BANG_HOSTDEVICE_FP16_DECL__ __half __int2half_rn__(const int i);
__BANG_HOSTDEVICE_FP16_DECL__ __half __uint2half_rn__(const unsigned int i);
__BANG_HOSTDEVICE_FP16_DECL__ __half __ll2half_rn(const long long int i);
__BANG_HOSTDEVICE_FP16_DECL__ __half __ull2half_rn(const unsigned long long int i);
__BANG_HOSTDEVICE_FP16_DECL__ unsigned int __half2uint_rz(const __half h);
__BANG_HOSTDEVICE_FP16_DECL__ int __half2int_rz(const __half h);
__BANG_HOSTDEVICE_FP16_DECL__ unsigned short int __half2ushort_rz(const __half h);
__BANG_HOSTDEVICE_FP16_DECL__ unsigned char __half2uchar_rz(const __half h);
__BANG_HOSTDEVICE_FP16_DECL__ signed char __half2char_rz(const __half h);

__BANG_HOSTDEVICE_FP16_DECL__ __half __hneg(const __half a);
__BANG_HOSTDEVICE_FP16_DECL__ bool __heq(const __half a, const __half b);
__BANG_HOSTDEVICE_FP16_DECL__ bool __hne(const __half a, const __half b);
__BANG_HOSTDEVICE_FP16_DECL__ bool __hle(const __half a, const __half b);
__BANG_HOSTDEVICE_FP16_DECL__ bool __hge(const __half a, const __half b);
__BANG_HOSTDEVICE_FP16_DECL__ bool __hlt(const __half a, const __half b);
__BANG_HOSTDEVICE_FP16_DECL__ bool __hgt(const __half a, const __half b);
__BANG_HOSTDEVICE_FP16_DECL__ bool __hneu(const __half a, const __half b);

__BANG_HOSTDEVICE_FP16_DECL__ __half __hadd(const __half a, const __half b);
__BANG_HOSTDEVICE_FP16_DECL__ __half __hsub(const __half a, const __half b);
__BANG_HOSTDEVICE_FP16_DECL__ __half __hmul(const __half a, const __half b);
__BANG_HOSTDEVICE_FP16_DECL__ __half __hdiv(const __half a, const __half b);

__BANG_HOSTDEVICE_FP16_DECL__ bool __hisnan(const __half a);

#if defined(__CPP_VERSION_AT_LEAST_11_FP16)
#else
    __BANG_HOSTDEVICE__ __half::__half() { }
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP16) */
__BANG_HOSTDEVICE__ __half &__half::operator=(const __half_raw &hr) {
    __x = hr.x;
    return *this;
}
__BANG_HOSTDEVICE__ volatile __half & __half::operator=(const __half_raw &hr) volatile { __x = hr.x; return *this; }
__BANG_HOSTDEVICE__ volatile __half & __half::operator=(const volatile __half_raw &hr) volatile { __x = hr.x; return *this; }
__BANG_HOSTDEVICE__ __half::operator __half_raw() const { __half_raw ret; ret.x = __x; return ret; }
__BANG_HOSTDEVICE__ __half::operator __half_raw() const volatile { __half_raw ret; ret.x = __x; return ret; }

__BANG_HOSTDEVICE__ __half::__half(const float f) { __x = __float2half__(f).__x; }
__BANG_HOSTDEVICE__ __half::operator float() const { return __half2float(*this); }
__BANG_HOSTDEVICE__ __half &__half::operator=(const float f) { __x = __float2half__(f).__x; return *this; }

__BANG_HOSTDEVICE__ __half::__half(const short val) { __x = __short2half_rn__(val).__x; }
__BANG_HOSTDEVICE__ __half::__half(const unsigned short val) { __x = __ushort2half_rn__(val).__x; }
__BANG_HOSTDEVICE__ __half::__half(const int val) { __x = __int2half_rn__(val).__x; }
__BANG_HOSTDEVICE__ __half::__half(const unsigned int val) { __x = __uint2half_rn__(val).__x; }

__BANG_HOSTDEVICE__ __half::__half(const long val) {

/* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER)
#pragma warning(disable : 4127)
#endif /* _MSC_VER */
    if (sizeof(long) == sizeof(long long))
#if defined(_MSC_VER)
#pragma warning(default : 4127)
#endif /* _MSC_VER */

    {
      __x = __ll2half_rn(static_cast<long long>(val)).__x;
    } else {
      __x = __int2half_rn__(static_cast<int>(val)).__x;
    }
}

__BANG_HOSTDEVICE__ __half::__half(const unsigned long val) {

/* Suppress VS warning: warning C4127: conditional expression is
 * constant */
#if defined(_MSC_VER)
#pragma warning(disable : 4127)
#endif /* _MSC_VER */
if (sizeof(unsigned long) == sizeof(unsigned long long))
#if defined(_MSC_VER)
#pragma warning(default : 4127)
#endif /* _MSC_VER */

{
    __x = __ull2half_rn(static_cast<unsigned long long>(val)).__x;
} else {
    __x = __uint2half_rn__(static_cast<unsigned int>(val)).__x;
}
}

__BANG_HOSTDEVICE__ __half::__half(const long long val) { __x = __ll2half_rn(val).__x; }
__BANG_HOSTDEVICE__ __half::__half(const unsigned long long val) { __x = __ull2half_rn(val).__x; }
__BANG_HOSTDEVICE__ __half::operator signed char() const { return __half2char_rz(*this); }
__BANG_HOSTDEVICE__ __half::operator unsigned char() const { return __half2uchar_rz(*this); }
__BANG_HOSTDEVICE__ __half::operator char() const {
    char value;
    /* Suppress VS warning: warning C4127: conditional expression is constant
     */
#if defined(_MSC_VER)
#pragma warning(disable : 4127)
#endif /* _MSC_VER */
    if (((char)-1) < (char)0)
#if defined(_MSC_VER)
#pragma warning(default : 4127)
#endif /* _MSC_VER */
    {
      value = static_cast<char>(__half2char_rz(*this));
    } else {
      value = static_cast<char>(__half2uchar_rz(*this));
    }
    return value;
}

__BANG_HOSTDEVICE__ __half::operator short() const { return __half2short_rz(*this); }
__BANG_HOSTDEVICE__ __half::operator unsigned short() const { return __half2ushort_rz(*this); }
__BANG_HOSTDEVICE__ __half::operator int() const { return __half2int_rz(*this); }
__BANG_HOSTDEVICE__ __half::operator unsigned int() const { return __half2uint_rz(*this); }

__BANG_HOSTDEVICE__ __half::operator long() const {
      long retval;
      /* Suppress VS warning: warning C4127: conditional expression is constant
       */
#if defined(_MSC_VER)
#pragma warning(disable : 4127)
#endif /* _MSC_VER */
      if (sizeof(long) == sizeof(long long))
#if defined(_MSC_VER)
#pragma warning(default : 4127)
#endif /* _MSC_VER */
      {
        retval = static_cast<long>(__half2ll_rz(*this));
      } else {
        retval = static_cast<long>(__half2int_rz(*this));
      }
      return retval;
}

__BANG_HOSTDEVICE__ __half::operator unsigned long() const {
      unsigned long retval;
/* Suppress VS warning: warning C4127: conditional expression is constant */
#if defined(_MSC_VER)
#pragma warning(disable : 4127)
#endif /* _MSC_VER */
      if (sizeof(unsigned long) == sizeof(unsigned long long))
#if defined(_MSC_VER)
#pragma warning(default : 4127)
#endif /* _MSC_VER */
      {
        retval = static_cast<unsigned long>(__half2ull_rz(*this));
      } else {
        retval = static_cast<unsigned long>(__half2uint_rz(*this));
      }
      return retval;
}

__BANG_HOSTDEVICE__ __half::operator long long() const { return __half2ll_rz(*this); }
__BANG_HOSTDEVICE__ __half::operator unsigned long long() const { return __half2ull_rz(*this); }
__BANG_HOSTDEVICE__ __half &__half::operator=(const short val) { __x = __short2half_rn__(val).__x; return *this; }
__BANG_HOSTDEVICE__ __half &__half::operator=(const unsigned short val) { __x = __ushort2half_rn__(val).__x; return *this; }
__BANG_HOSTDEVICE__ __half &__half::operator=(const int val) { __x = __int2half_rn__(val).__x; return *this; }
__BANG_HOSTDEVICE__ __half &__half::operator=(const unsigned int val) { __x = __uint2half_rn__(val).__x; return *this; }
__BANG_HOSTDEVICE__ __half &__half::operator=(const long long val) { __x = __ll2half_rn(val).__x; return *this; }
__BANG_HOSTDEVICE__ __half &__half::operator=(const long val) { __x = __half(val).__x; return *this; }
__BANG_HOSTDEVICE__ __half &__half::operator=(const unsigned long val) { __x = __half(val).__x; return *this; }
__BANG_HOSTDEVICE__ __half &__half::operator=(const unsigned long long val) { __x = __ull2half_rn(val).__x; return *this; }

__BANG_HOSTDEVICE__ __half operator+(const __half &lh, const __half &rh) { return __hadd(lh, rh); }
__BANG_HOSTDEVICE__ __half operator-(const __half &lh, const __half &rh) { return __hsub(lh, rh); }
__BANG_HOSTDEVICE__ __half operator*(const __half &lh, const __half &rh) { return __hmul(lh, rh); }
__BANG_HOSTDEVICE__ __half operator/(const __half &lh, const __half &rh) { return __hdiv(lh, rh); }
__BANG_HOSTDEVICE__ __half &operator+=(__half &lh, const __half &rh) { lh = __hadd(lh, rh); return lh; }
__BANG_HOSTDEVICE__ __half &operator-=(__half &lh, const __half &rh) { lh = __hsub(lh, rh); return lh; }
__BANG_HOSTDEVICE__ __half &operator*=(__half &lh, const __half &rh) { lh = __hmul(lh, rh); return lh; }
__BANG_HOSTDEVICE__ __half &operator/=(__half &lh, const __half &rh) { lh = __hdiv(lh, rh); return lh; }
__BANG_HOSTDEVICE__ __half &operator++(__half &h)      { __half_raw one; one.x = 0x3C00U; h += one; return h; }
__BANG_HOSTDEVICE__ __half &operator--(__half &h)      { __half_raw one; one.x = 0x3C00U; h -= one; return h; }
__BANG_HOSTDEVICE__ __half  operator++(__half &h, const int ignored) {
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __half ret = h;
    __half_raw one;
    one.x = 0x3C00U;
    h += one;
    return ret;
}

__BANG_HOSTDEVICE__ __half  operator--(__half &h, const int ignored)
{
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __half ret = h;
    __half_raw one;
    one.x = 0x3C00U;
    h -= one;
    return ret;
}

__BANG_HOSTDEVICE__ __half operator+(const __half &h) { return h; }
__BANG_HOSTDEVICE__ __half operator-(const __half &h) { return __hneg(h); }
__BANG_HOSTDEVICE__ bool operator==(const __half &lh, const __half &rh) { return __heq(lh, rh); }
__BANG_HOSTDEVICE__ bool operator!=(const __half &lh, const __half &rh) { return __hneu(lh, rh); }
__BANG_HOSTDEVICE__ bool operator> (const __half &lh, const __half &rh) { return __hgt(lh, rh); }
__BANG_HOSTDEVICE__ bool operator< (const __half &lh, const __half &rh) { return __hlt(lh, rh); }
__BANG_HOSTDEVICE__ bool operator>=(const __half &lh, const __half &rh) { return __hge(lh, rh); }
__BANG_HOSTDEVICE__ bool operator<=(const __half &lh, const __half &rh) { return __hle(lh, rh); }

/* Restore warning for multiple assignment operators */
#if defined(_MSC_VER) && _MSC_VER >= 1500
#pragma warning( pop )
#endif /* defined(_MSC_VER) && _MSC_VER >= 1500 */

/* Restore -Weffc++ warnings from here on */
#if defined(__GNUC__)
#if __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6)
#pragma GCC diagnostic pop
#endif /* __GNUC__ > 4 || (__GNUC__ == 4 && __GNUC_MINOR__ >= 6) */
#endif /* defined(__GNUC__) */

#undef __BANG_HOSTDEVICE__
#undef __BANG_ALIGN__

static inline unsigned short __internal_float2half(const float f, unsigned int &sign, unsigned int &remainder)
{
    unsigned int u;
    unsigned int result;
    unsigned int x = *(reinterpret_cast<const unsigned int *>(&(f)));
    u = (x & 0x7fffffffU);
    sign = ((x >> 16U) & 0x8000U);
    // NaN/+Inf/-Inf
    if (u >= 0x7f800000U) {
        remainder = 0U;
        result = ((u == 0x7f800000U) ? (sign | 0x7c00U) : 0x7fffU);
    } else if (u > 0x477fefffU) { // Overflows
        remainder = 0x80000000U;
        result = (sign | 0x7bffU);
    } else if (u >= 0x38800000U) { // Normal numbers
        remainder = u << 19U;
        u -= 0x38000000U;
        result = (sign | (u >> 13U));
    } else if (u < 0x33000001U) { // +0/-0
        remainder = u;
        result = sign;
    } else { // Denormal numbers
        const unsigned int exponent = u >> 23U;
        const unsigned int shift = 0x7eU - exponent;
        unsigned int mantissa = (u & 0x7fffffU);
        mantissa |= 0x800000U;
        remainder = mantissa << (32U - shift);
        result = (sign | (mantissa >> shift));
        result &= 0x0000FFFFU;
    }
    return static_cast<unsigned short>(result);
}

static inline float __internal_half2float(const unsigned short h) {
    unsigned int sign = ((static_cast<unsigned int>(h) >> 15U) & 1U);
    unsigned int exponent = ((static_cast<unsigned int>(h) >> 10U) & 0x1fU);
    unsigned int mantissa = ((static_cast<unsigned int>(h) & 0x3ffU) << 13U);
    float f;
    if (exponent == 0x1fU) { /* NaN or Inf */
        /* discard sign of a NaN */
        sign = ((mantissa != 0U) ? (sign >> 1U) : sign);
        mantissa = ((mantissa != 0U) ? 0x7fffffU : 0U);
        exponent = 0xffU;
    } else if (exponent == 0U) { /* Denorm or Zero */
        if (mantissa != 0U) {
          unsigned int msb;
          exponent = 0x71U;
          do {
            msb = (mantissa & 0x400000U);
            mantissa <<= 1U; /* normalize */
            --exponent;
          } while (msb == 0U);
          mantissa &= 0x7fffffU; /* 1.mantissa is implicit */
        }
    } else {
        exponent += 0x70U;
    }
    const unsigned int u = ((sign << 31U) | (exponent << 23U) | mantissa);
    f = *(reinterpret_cast<const float *>(&(u)));
    return f;
}

__BANG_HOSTDEVICE_FP16_DECL__ __half __float2half__(const float a) {
    __half val;
#ifdef __BANG_ARCH__
    asm("cvt.gpr.rn.f16.f32 %0, %1;" : "+r"(__HALF_TO_US(val)) : "r"(a));
#else
    __half_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2half(a, sign, remainder);
    if ((remainder > 0x80000000U) ||
        ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U))) {
        r.x++;
    }
    val = r;
#endif
    return val;
}

__BANG_HOSTDEVICE_FP16_DECL__ float __half2float(const __half a) {
    float val;
#ifdef __BANG_ARCH__
    asm("cvt.gpr.f32.f16 %0, %1;" : "+r"(val) : "r"(__HALF_TO_CUS(a)));
#else
    val = __internal_half2float(static_cast<__half_raw>(a).x);
#endif
    return val;
}

__BANG_HOSTDEVICE_FP16_DECL__ __half __float2half_rn__(const float a) {
    __half val;
#ifdef __BANG_ARCH__
    asm("cvt.gpr.rn.f16.f32 %0, %1;" : "+r"(__HALF_TO_US(val)) : "r"(a));
#else
    __half_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2half(a, sign, remainder);
    if ((remainder > 0x80000000U) ||
        ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U))) {
        r.x++;
    }
    val = r;
#endif
    return val;
}

__BANG_HOSTDEVICE_FP16_DECL__ __half __short2half_rn__(const short int i) {
    __half h;
#ifdef __BANG_ARCH__
    asm("cvt.gpr.rn.f16.s16 %0, %1;" : "+r"(__HALF_TO_US(h)) : "r"(i));
#else
    const float f = static_cast<float>(i);
    h = __float2half_rn__(f);
#endif
    return h;
}

__BANG_HOSTDEVICE_FP16_DECL__ __half __int2half_rn__(const int i) {
    __half h;
#ifdef __BANG_ARCH__
    asm("cvt.gpr.rn.f16.s32 %0, %1;" : "+r"(__HALF_TO_US(h)) : "r"(i));
#else
    // double-rounding is not a problem here: if integer
    // has more than 24 bits, it is already too large to
    // be represented in half precision, and result will
    // be infinity.
    const float f = static_cast<float>(i);
    h = __float2half_rn__(f);
#endif
    return h;
}

__BANG_HOSTDEVICE_FP16_DECL__ __half __uint2half_rn__(const unsigned int i) {
    __half h;
#ifdef __BANG_ARCH__
    asm("cvt.gpr.rn.f16.u32 %0, %1;" : "+r"(__HALF_TO_US(h)) : "r"(i));
#else
    // double-rounding is not a problem here: if integer
    // has more than 24 bits, it is already too large to
    // be represented in half precision, and result will
    // be infinity.
    const float f = static_cast<float>(i);
    h = __float2half_rn__(f);
#endif
    return h;
}

__BANG_HOSTDEVICE_FP16_DECL__ long long int __half2ll_rz(const __half h) {
    long long int i;
#ifdef __BANG_ARCH__
    asm("cvt.gpr.tz.s64.f16 %0, %1;" : "+r"(i) : "r"(__HALF_TO_CUS(h)));
#else
    const float f = __half2float(h);
    const long long int max_val = (long long int)0x7fffffffffffffffULL;
    const long long int min_val = (long long int)0x8000000000000000ULL;
    const unsigned short bits =
        static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xF800U) {
        // NaN
        i = min_val;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<long long int>(f);
    }
#endif
    return i;
}

__BANG_HOSTDEVICE_FP16_DECL__ unsigned long long int __half2ull_rz(const __half h) {
    unsigned long long int i;

#ifdef __BANG_ARCH__
    asm("cvt.gpr.tz.u64.f16 %0, %1;" : "+r"(i) : "r"(__HALF_TO_CUS(h)));
#else
    const float f = __half2float(h);
    const unsigned long long int max_val = 0xffffffffffffffffULL;
    const unsigned long long int min_val = 0ULL;
    const unsigned short bits =
        static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xF800U) {
        // NaN
        i = 0x8000000000000000ULL;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<unsigned long long int>(f);
    }
#endif
    return i;
}

__BANG_HOSTDEVICE_FP16_DECL__ __half __ushort2half_rn__(const unsigned short int i) {
    __half h;
#ifdef __BANG_ARCH__
    asm("cvt.gpr.rn.f16.u16 %0, %1;" : "+r"(__HALF_TO_US(h)) : "r"(i));
#else
    const float f = static_cast<float>(i);
    h = __float2half_rn__(f);
#endif

    return h;
}

__BANG_HOSTDEVICE_FP16_DECL__ __half __ll2half_rn(const long long int i) {
    __half h;
#ifdef __BANG_ARCH__
    asm("cvt.gpr.rn.f16.s64 %0, %1;" : "+r"(__HALF_TO_US(h)) : "r"(i));
#else
    // double-rounding is not a problem here: if integer
    // has more than 24 bits, it is already too large to
    // be represented in half precision, and result will
    // be infinity.
    const float f = static_cast<float>(i);
    h = __float2half_rn__(f);
#endif
    return h;
}

__BANG_HOSTDEVICE_FP16_DECL__ __half __ull2half_rn(const unsigned long long int i) {
    __half h;
#ifdef __BANG_ARCH__
    asm("cvt.gpr.rn.f16.u64 %0, %1;" : "+r"(__HALF_TO_US(h)) : "r"(i));
#else
    // double-rounding is not a problem here: if integer
    // has more than 24 bits, it is already too large to
    // be represented in half precision, and result will
    // be infinity.
    const float f = static_cast<float>(i);
    h = __float2half_rn__(f);
#endif
    return h;
}

__BANG_HOSTDEVICE_FP16_DECL__ unsigned int __half2uint_rz(const __half h) {
    unsigned int i;

#ifdef __BANG_ARCH__
    asm("cvt.gpr.tz.u32.f16 %0, %1;" : "=r"(i) : "r"(__HALF_TO_CUS(h)));
#else
    const float f = __half2float(h);
    const unsigned int max_val = 0xffffffffU;
    const unsigned int min_val = 0U;
    const unsigned short bits =
        static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xF800U) {
        // NaN
        i = 0U;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<unsigned int>(f);
    }
#endif
    return i;
}

__BANG_HOSTDEVICE_FP16_DECL__ int __half2int_rz(const __half h) {
    int i;
#ifdef __BANG_ARCH__
    asm("cvt.gpr.tz.s32.f16 %0, %1;" : "=r"(i) : "r"(__HALF_TO_CUS(h)));
#else
    const float f = __half2float(h);
    const int max_val = (int)0x7fffffffU;
    const int min_val = (int)0x80000000U;
    const unsigned short bits =
        static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xF800U) {
        // NaN
        i = 0;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<int>(f);
    }
#endif
    return i;
}

__BANG_HOSTDEVICE_FP16_DECL__ unsigned short int __half2ushort_rz(const __half h) {
    unsigned short int i;

#ifdef __BANG_ARCH__
    asm("cvt.gpr.tz.u16.f16 %0, %1;" : "+r"(i) : "r"(__HALF_TO_CUS(h)));
#else
    const float f = __half2float(h);
    const unsigned short int max_val = 0xffffU;
    const unsigned short int min_val = 0U;
    const unsigned short bits =
        static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xF800U) {
        // NaN
        i = 0U;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<unsigned short int>(f);
    }
#endif
    return i;
}

__BANG_HOSTDEVICE_FP16_DECL__ unsigned char __half2uchar_rz(const __half h) {
    unsigned char i;

#ifdef __BANG_ARCH__
    unsigned int tmp;
    asm("cvt.gpr.tz.u8.f16 %0, %1;" : "=r"(tmp) : "r"(__HALF_TO_CUS(h)));
    i = static_cast<unsigned char>(tmp);
#else
    const float f = __half2float(h);
    const unsigned char max_val = 0xffU;
    const unsigned char min_val = 0U;
    const unsigned short bits =
        static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xF800U) {
        // NaN
        i = 0U;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<unsigned char>(f);
    }
#endif
    return i;
}

__BANG_HOSTDEVICE_FP16_DECL__ signed char __half2char_rz(const __half h) {
    signed char i;
#ifdef __BANG_ARCH__
    unsigned int tmp;
    asm("cvt.gpr.tz.s8.f16 %0, %1;" : "=r"(tmp) : "r"(__HALF_TO_CUS(h)));
    const unsigned char u = static_cast<unsigned char>(tmp);
    i = static_cast<signed char>(u);
#else
    const float f = __half2float(h);
    const signed char max_val = (signed char)0x7fU;
    const signed char min_val = (signed char)0x80U;
    const unsigned short bits =
        static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xF800U) {
        // NaN
        i = 0;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<signed char>(f);
    }
#endif
    return i;
}

__BANG_HOSTDEVICE_FP16_DECL__ short int __half2short_rz(const __half h) {
    short int i;
#ifdef __BANG_ARCH__
    asm("cvt.gpr.tz.s16.f16 %0, %1;" : "+r"(i) : "r"(__HALF_TO_CUS(h)));
#else
    const float f = __half2float(h);
    const short int max_val = (short int)0x7fffU;
    const short int min_val = (short int)0x8000U;
    const unsigned short bits =
        static_cast<unsigned short>(static_cast<__half_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xF800U) {
        // NaN
        i = 0;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        // normal value, conversion is well-defined
        i = static_cast<short int>(f);
    }
#endif
    return i;
}

/******************************************************************************
*                             __half arithmetic                             *
******************************************************************************/
#define ___BANG_FP16_STRINGIFY_INNERMOST(x) #x
#define __BANG_FP16_STRINGIFY(x) ___BANG_FP16_STRINGIFY_INNERMOST(x)

/* Macros for half binary arithmetic */
#define __BINARY_OP_HALF_MACRO(name) /* do */ {\
   __half val; \
   asm(__BANG_FP16_STRINGIFY(name) ".gpr.f16 %0, %1, %2;" \
        :"+r"(__HALF_TO_US(val)) : "r"(__HALF_TO_CUS(a)),"r"(__HALF_TO_CUS(b))); \
   return val; \
} /* while(0) */

__BANG_HOSTDEVICE_FP16_DECL__ __half __hadd(const __half a, const __half b) {
#ifdef __BANG_ARCH__
    __BINARY_OP_HALF_MACRO(add);
#else
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return __float2half__(fa + fb);
#endif
}

__BANG_HOSTDEVICE_FP16_DECL__ __half __hsub(const __half a, const __half b) {
#ifdef __BANG_ARCH__
    __BINARY_OP_HALF_MACRO(sub);
#else
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return __float2half__(fa - fb);
#endif
}

__BANG_HOSTDEVICE_FP16_DECL__ __half __hmul(const __half a, const __half b) {
#ifdef __BANG_ARCH__
    __BINARY_OP_HALF_MACRO(mul);
#else
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return __float2half__(fa * fb);
#endif
}

__BANG_HOSTDEVICE_FP16_DECL__ __half __hdiv(const __half a, const __half b) {
#ifdef __BANG_ARCH__
    __BINARY_OP_HALF_MACRO(div);
#else
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return __float2half__(fa / fb);
#endif
}

__BANG_HOSTDEVICE_FP16_DECL__ __half __hneg(const __half a) {
#ifdef __BANG_ARCH__
    __half r;
    asm("neg.gpr.f16 %0, %1;" :"+r"(__HALF_TO_US(r)) : "r"(__HALF_TO_CUS(a)));
    return r;
#else
    const float fa = __half2float(a);
    return __float2half__(-fa);
#endif
}

/******************************************************************************
*                             __half comparison                              *
******************************************************************************/
#define __COMPARISON_OP_HALF_MACRO(name, a, b) /* do */ {\
   char val; \
   asm( "{ .gpr %%__$temp3;\n" \
        "  " __BANG_FP16_STRINGIFY(name) ".gpr.f16  %%__$temp3, %1, %2;\n" \
        "  cvt.gpr.tz.s8.f16 %0, %%__$temp3;}" \
        : "+r"(val) : "r"(__HALF_TO_CUS(a)), "r"(__HALF_TO_CUS(b))); \
   return val; \
} /* while(0) */

__BANG_HOSTDEVICE_FP16_DECL__ bool __heq(const __half a, const __half b) {
#ifdef __BANG_ARCH__
    __COMPARISON_OP_HALF_MACRO(eq, a, b);
#else
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa == fb);
#endif
}

__BANG_HOSTDEVICE_FP16_DECL__ bool __hne(const __half a, const __half b) {
#ifdef __BANG_ARCH__
   __COMPARISON_OP_HALF_MACRO(ne, a, b);
#else
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa != fb) && (!__hisnan(a)) && (!__hisnan(b));
#endif
}

__BANG_HOSTDEVICE_FP16_DECL__ bool __hle(const __half a, const __half b) {
#ifdef __BANG_ARCH__
   __COMPARISON_OP_HALF_MACRO(ge, b, a);
#else
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa <= fb);
#endif
}

__BANG_HOSTDEVICE_FP16_DECL__ bool __hge(const __half a, const __half b) {
#ifdef __BANG_ARCH__
   __COMPARISON_OP_HALF_MACRO(ge, a, b);
#else
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa >= fb);
#endif
}

__BANG_HOSTDEVICE_FP16_DECL__ bool __hlt(const __half a, const __half b) {
#ifdef __BANG_ARCH__
   __COMPARISON_OP_HALF_MACRO(lt, a, b);
#else
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa < fb);
#endif
}

__BANG_HOSTDEVICE_FP16_DECL__ bool __hgt(const __half a, const __half b) {
#ifdef __BANG_ARCH__
   __COMPARISON_OP_HALF_MACRO(lt, b, a);
#else
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa > fb);
#endif
}

__BANG_HOSTDEVICE_FP16_DECL__ bool __hneu(const __half a, const __half b) {
#ifdef __BANG_ARCH__
   __COMPARISON_OP_HALF_MACRO(neu, a, b);
#else
    const float fa = __half2float(a);
    const float fb = __half2float(b);
    return (fa != fb);
#endif
}

__BANG_HOSTDEVICE_FP16_DECL__ bool __hisnan(const __half a) {
#if (__BANG_ARCH__ >= 600)
    bool r;
    asm("test.nan.gpr.f16 %0, %1;" : "+r"(r) : "r"(__HALF_TO_CUS(a)));
    return r;
#else
    const __half_raw hr = static_cast<__half_raw>(a);
    return ((hr.x & (unsigned short)0x7FFFU) > (unsigned short)0x7C00U);
#endif
}

#undef __BINARY_OP_HALF_MACRO
#undef ___BANG_FP16_STRINGIFY_INNERMOST
#undef __BANG_FP16_STRINGIFY
#undef __COMPARISON_OP_HALF_MACRO

#undef __BANG_HOSTDEVICE_FP16_DECL__
#undef __BANG_FP16_DECL__

#undef __HALF_TO_US
#undef __HALF_TO_CUS
#undef __BANG_FP16_CONSTEXPR__

#if defined(__CPP_VERSION_AT_LEAST_11_FP16)
#undef __CPP_VERSION_AT_LEAST_11_FP16
#endif /* defined(__CPP_VERSION_AT_LEAST_11_FP16) */

#endif /* end of include guard: __BANG_FP16_HPP__ */
