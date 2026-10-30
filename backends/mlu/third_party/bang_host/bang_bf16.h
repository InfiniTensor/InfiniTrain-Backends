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
#if !defined(__BANG_BF16_H__)
#define __BANG_BF16_H__

#include "bang_fp16.h"

#if defined(__BANGCC__)
#define __BANG_HOSTDEVICE_BF16__ __mlu_host__ __mlu_func__
#define __BANG_HOSTDEVICE_BF16_DECL__ static __mlu_host__ __mlu_func__
#else /* !defined(__BANGCC__) */
#if defined(__GNUC__)
#define __BANG_HOSTDEVICE_BF16_DECL__ static __attribute__((unused))
#else
#define __BANG_HOSTDEVICE_BF16_DECL__ static
#endif /* defined(__GNUC__) */
#define __BANG_HOSTDEVICE_BF16__ inline
#endif /* defined(__BANGCC_) */

#if !defined(_MSC_VER) && __cplusplus >= 201103L
#   define __CPP_VERSION_AT_LEAST_11_BF16
#elif _MSC_FULL_VER >= 190024210 && _MSVC_LANG >= 201103L
#   define __CPP_VERSION_AT_LEAST_11_BF16
#endif

/* Set up structure-alignment attribute */
#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
#define __BANG_ALIGN_BF16__(n) alignas(n)    /* C++11 kindly gives us a keyword for this */
#define __BANG_BF16_CONSTEXPR__ constexpr
#else /* defined(__CPP_VERSION_AT_LEAST_11_BF16)*/
#if defined(__GNUC__)
#define __BANG_ALIGN_BF16__(n) __attribute__ ((aligned(n)))
#elif defined(_MSC_VER)
#define __BANG_ALIGN_BF16__(n) __declspec(align(n))
#else
#define __BANG_ALIGN_BF16__(n)
#endif /* defined(__GNUC__) */
#define __BANG_BF16_CONSTEXPR__
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) */

/* Macros to allow bang_bfloat16  to be used by inline assembly */
#define __BFLOAT16_TO_US(var) *(reinterpret_cast<unsigned short *>(&(var)))
#define __BFLOAT16_TO_CUS(var) *(reinterpret_cast<const unsigned short *>(&(var)))
#define __FLOAT_TO_UINT(var) *(reinterpret_cast<unsigned int *>(&(var)))
#define __UINT_TO_FLOAT(var) *(reinterpret_cast<const unsigned short *>(&(var)))

typedef struct __BANG_ALIGN_BF16__(2) {
    unsigned short x;
} __bang_bfloat16_raw;

/* All other definitions in this file are only visible to C++ compilers */
#if defined(__cplusplus)

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
#endif /* defined(__GNUC__) */

struct __BANG_ALIGN_BF16__(2) __bang_bfloat16;
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __float2bfloat16__(const float a);
__BANG_HOSTDEVICE_BF16_DECL__ float __bfloat162float__(const __bang_bfloat16 a);
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __short2bfloat16_rn__(const short int i);
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __ushort2bfloat16_rn__(const unsigned short int i);
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __int2bfloat16_rn__(const int i);
__BANG_HOSTDEVICE_BF16_DECL__ int __bfloat162int_rz__(const __bang_bfloat16 h);
__BANG_HOSTDEVICE_BF16_DECL__ short int __bfloat162short_rz__(const __bang_bfloat16 h);
__BANG_HOSTDEVICE_BF16_DECL__ unsigned int __bfloat162uint_rz__(const __bang_bfloat16 h);
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __uint2bfloat16_rn__(const unsigned int i);
__BANG_HOSTDEVICE_BF16_DECL__ unsigned short int __bfloat162ushort_rz__(const __bang_bfloat16 h);

__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __hneg(const __bang_bfloat16 a);
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __hsub(const __bang_bfloat16 a, const __bang_bfloat16 b);
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __hadd(const __bang_bfloat16 a, const __bang_bfloat16 b);
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __hmul(const __bang_bfloat16 a, const __bang_bfloat16 b);
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __hdiv(const __bang_bfloat16 a, const __bang_bfloat16 b);
__BANG_HOSTDEVICE_BF16_DECL__ bool __heq(const __bang_bfloat16 a, const __bang_bfloat16 b);
__BANG_HOSTDEVICE_BF16_DECL__ bool __hneu(const __bang_bfloat16 a, const __bang_bfloat16 b);
__BANG_HOSTDEVICE_BF16_DECL__ bool __hgt(const __bang_bfloat16 a, const __bang_bfloat16 b);
__BANG_HOSTDEVICE_BF16_DECL__ bool __hlt(const __bang_bfloat16 a, const __bang_bfloat16 b);
__BANG_HOSTDEVICE_BF16_DECL__ bool __hge(const __bang_bfloat16 a, const __bang_bfloat16 b);
__BANG_HOSTDEVICE_BF16_DECL__ bool __hle(const __bang_bfloat16 a, const __bang_bfloat16 b);

__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __ll2bfloat16_rn(const long long int i);
__BANG_HOSTDEVICE_BF16_DECL__ long long int __bfloat162ll_rz(const __bang_bfloat16 h);
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __ull2bfloat16_rn(const unsigned long long int i);
__BANG_HOSTDEVICE_BF16_DECL__ unsigned long long int __bfloat162ull_rz(const __bang_bfloat16 h);

struct __BANG_ALIGN_BF16__(2) __bang_bfloat16 {
protected:
    unsigned short __x;

public:
#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
    __bang_bfloat16() = default;
#else
    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16() { }
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) */

    /* Convert to/from __bang_bfloat16_raw */
    __BANG_HOSTDEVICE_BF16__ __BANG_BF16_CONSTEXPR__ __bang_bfloat16(const __bang_bfloat16_raw &hr) : __x(hr.x) { }
    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16 &operator=(const __bang_bfloat16_raw &hr) { __x = hr.x; return *this; }
    __BANG_HOSTDEVICE_BF16__ volatile __bang_bfloat16 &operator=(const __bang_bfloat16_raw &hr) volatile { __x = hr.x; return *this; }
    __BANG_HOSTDEVICE_BF16__ volatile __bang_bfloat16 &operator=(const volatile __bang_bfloat16_raw &hr) volatile { __x = hr.x; return *this; }
    __BANG_HOSTDEVICE_BF16__ operator __bang_bfloat16_raw() const { __bang_bfloat16_raw ret; ret.x = __x; return ret; }
    __BANG_HOSTDEVICE_BF16__ operator __bang_bfloat16_raw() const volatile { __bang_bfloat16_raw ret; ret.x = __x; return ret; }

    // /* Construct from float */
    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16(const float f) { __x = __float2bfloat16__(f).__x;  }

    __BANG_HOSTDEVICE_BF16__ operator float() const { return __bfloat162float__(*this); }
    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16 &operator=(const float f) { __x = __float2bfloat16__(f).__x; return *this; }

    // /* Allow automatic construction from types supported natively in hardware */
    // /* Note we do avoid constructor init-list because of special host/device compilation rules */
    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16(short val) { __x = __short2bfloat16_rn__(val).__x;  }
    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16(unsigned short val) { __x = __ushort2bfloat16_rn__(val).__x;  }
    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16(int val) { __x = __int2bfloat16_rn__(val).__x;  }
    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16(unsigned int val) { __x = __uint2bfloat16_rn__(val).__x;  }

    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16(const __half f) {
#if defined(__BANG_ARCH__)
      asm("cvt.gpr.rn.bf16.f16 %0, %1;" : "+r"(__x) : "r"(__BFLOAT16_TO_CUS(f)));
#else
      __x = __float2bfloat16__(__half2float(f)).__x;
#endif
    }

    /* Allow automatic casts to supported builtin types, matching all that are permitted with float */
    __BANG_HOSTDEVICE_BF16__ operator short() const { return __bfloat162short_rz__(*this); }
    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16 &operator=(short val) { __x = __short2bfloat16_rn__(val).__x; return *this; }

    __BANG_HOSTDEVICE_BF16__ operator unsigned short() const { return __bfloat162ushort_rz__(*this); }
    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16 &operator=(unsigned short val) { __x = __ushort2bfloat16_rn__(val).__x; return *this; }

    __BANG_HOSTDEVICE_BF16__ operator int() const { return __bfloat162int_rz__(*this); }
    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16 &operator=(int val) { __x = __int2bfloat16_rn__(val).__x; return *this; }

    __BANG_HOSTDEVICE_BF16__ operator unsigned int() const { return __bfloat162uint_rz__(*this); }
    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16 &operator=(unsigned int val) { __x = __uint2bfloat16_rn__(val).__x; return *this; }

    /* Boolean conversion - note both 0 and -0 must return false */
    __BANG_HOSTDEVICE_BF16__ operator bool() const { return (__x & 0x7FFF) != 0; }

    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16(const long val) {
      if (sizeof(long) == sizeof(long long)) {
        __x = __ll2bfloat16_rn(static_cast<long long>(val)).__x;
      } else {
        __x = __int2bfloat16_rn__(static_cast<int>(val)).__x;
      }
    }

    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16(const unsigned long val) {
      if (sizeof(unsigned long) == sizeof(unsigned long long)) {
        __x = __ull2bfloat16_rn(static_cast<unsigned long long>(val)).__x;
      } else {
        __x = __uint2bfloat16_rn__(static_cast<unsigned int>(val)).__x;
      }
    }

    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16(long long val) { __x = __ll2bfloat16_rn(val).__x;  }
    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16(unsigned long long val) { __x = __ull2bfloat16_rn(val).__x; }

    __BANG_HOSTDEVICE_BF16__ operator long() const {
      long retval;
      if (sizeof(long) == sizeof(long long)) {
        retval = static_cast<long>(__bfloat162ll_rz(*this));
      } else {
        retval = static_cast<long>(__bfloat162int_rz__(*this));
      }
      return retval;
    }

    __BANG_HOSTDEVICE_BF16__ operator unsigned long() const {
      unsigned long retval;
      if (sizeof(unsigned long) == sizeof(unsigned long long)) {
        retval = static_cast<unsigned long>(__bfloat162ull_rz(*this));
      } else {
        retval = static_cast<unsigned long>(__bfloat162uint_rz__(*this));
      }
      return retval;
    }

    __BANG_HOSTDEVICE_BF16__ operator long long() const { return __bfloat162ll_rz(*this); }
    __BANG_HOSTDEVICE_BF16__ operator unsigned long long() const { return __bfloat162ull_rz(*this); }

    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16 &operator=(long long val) { __x = __ll2bfloat16_rn(val).__x; return *this; }
    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16 &operator=(long val) { __x = __bang_bfloat16(val).__x; return *this; }

    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16 &operator=(unsigned long long val) { __x = __ull2bfloat16_rn(val).__x; return *this; }
    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16 &operator=(unsigned long val) { __x = __bang_bfloat16(val).__x; return *this; }

    __BANG_HOSTDEVICE_BF16__ __bang_bfloat16 &operator=(__half f) { __x = __bang_bfloat16(f).__x; return *this; }
};

/* Some basic arithmetic operations expected of a builtin */
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 operator+(const __bang_bfloat16 &lh, const __bang_bfloat16 &rh) { return __hadd(lh, rh); }
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 operator-(const __bang_bfloat16 &lh, const __bang_bfloat16 &rh) { return __hsub(lh, rh); }
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 operator*(const __bang_bfloat16 &lh, const __bang_bfloat16 &rh) { return __hmul(lh, rh); }
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 operator/(const __bang_bfloat16 &lh, const __bang_bfloat16 &rh) { return __hdiv(lh, rh); }

__BANG_HOSTDEVICE_BF16_DECL__  __bang_bfloat16 &operator+=(__bang_bfloat16 &lh, const __bang_bfloat16 &rh) { lh = __hadd(lh, rh); return lh; }
__BANG_HOSTDEVICE_BF16_DECL__  __bang_bfloat16 &operator-=(__bang_bfloat16 &lh, const __bang_bfloat16 &rh) { lh = __hsub(lh, rh); return lh; }
__BANG_HOSTDEVICE_BF16_DECL__  __bang_bfloat16 &operator*=(__bang_bfloat16 &lh, const __bang_bfloat16 &rh) { lh = __hmul(lh, rh); return lh; }
__BANG_HOSTDEVICE_BF16_DECL__  __bang_bfloat16 &operator/=(__bang_bfloat16 &lh, const __bang_bfloat16 &rh) { lh = __hdiv(lh, rh); return lh; }

/* Note for increment and decrement we use the raw value 0x3F80 equating to bang_bfloat16(1.0f), to avoid the extra conversion */
__BANG_HOSTDEVICE_BF16_DECL__  __bang_bfloat16 &operator++(__bang_bfloat16 &h)      { __bang_bfloat16_raw one; one.x = 0x3F80; h += one; return h; }
__BANG_HOSTDEVICE_BF16_DECL__  __bang_bfloat16 &operator--(__bang_bfloat16 &h)      { __bang_bfloat16_raw one; one.x = 0x3F80; h -= one; return h; }
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 operator++(__bang_bfloat16 &h, const int ignored) {
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __bang_bfloat16 ret = h;
    __bang_bfloat16_raw one;
    one.x = 0x3F80;
    h += one;
    return ret;
}
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16  operator--(__bang_bfloat16 &h, const int ignored)
{
    // ignored on purpose. Parameter only needed to distinguish the function declaration from other types of operators.
    static_cast<void>(ignored);

    const __bang_bfloat16 ret = h;
    __bang_bfloat16_raw one;
    one.x = 0x3F80;
    h -= one;
    return ret;
}
/* Unary plus and inverse operators */
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 operator+(const __bang_bfloat16 &h) { return h; }
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 operator-(const __bang_bfloat16 &h) { return __hneg(h); }

/* Some basic comparison operations to make it look like a builtin */
__BANG_HOSTDEVICE_BF16_DECL__ bool operator==(const __bang_bfloat16 &lh, const __bang_bfloat16 &rh) { return __heq(lh, rh); }
__BANG_HOSTDEVICE_BF16_DECL__ bool operator!=(const __bang_bfloat16 &lh, const __bang_bfloat16 &rh) { return __hneu(lh, rh); }
__BANG_HOSTDEVICE_BF16_DECL__ bool operator> (const __bang_bfloat16 &lh, const __bang_bfloat16 &rh) { return __hgt(lh, rh); }
__BANG_HOSTDEVICE_BF16_DECL__ bool operator< (const __bang_bfloat16 &lh, const __bang_bfloat16 &rh) { return __hlt(lh, rh); }
__BANG_HOSTDEVICE_BF16_DECL__ bool operator>=(const __bang_bfloat16 &lh, const __bang_bfloat16 &rh) { return __hge(lh, rh); }
__BANG_HOSTDEVICE_BF16_DECL__ bool operator<=(const __bang_bfloat16 &lh, const __bang_bfloat16 &rh) { return __hle(lh, rh); }

__BANG_HOSTDEVICE_BF16__ __half::__half(const __bang_bfloat16 f) {
#if defined(__BANG_ARCH__)
    asm("cvt.gpr.rn.f16.bf16 %0, %1;" : "+r"(__x) : "r"(__BFLOAT16_TO_CUS(f)));
#else
    __x = __float2half_rn__(__bfloat162float__(f)).__x;
#endif
}

__BANG_HOSTDEVICE_BF16__ __half &__half::operator=(const __bang_bfloat16 f) { __x = __half(f).__x; return *this; }

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

#undef __BANG_HOSTDEVICE_BF16__
#undef __BANG_ALIGN_BF16__
#undef __BANG_BF16_CONSTEXPR__

__BANG_HOSTDEVICE_BF16_DECL__ unsigned int __internal_float_as_uint(const float f) {
  unsigned int u = *(reinterpret_cast<const unsigned int *>(&(f)));
  return u;
}

__BANG_HOSTDEVICE_BF16_DECL__ float __internal_uint_as_float(unsigned int u) {
  float f = *(reinterpret_cast<float *>(&u));
  return f;
}

__BANG_HOSTDEVICE_BF16_DECL__ unsigned short __internal_float2bfloat16(const float f, unsigned int &sign, unsigned int &remainder)
{
    unsigned int x;

    x = __internal_float_as_uint(f);

    if ((x & 0x7fffffffU) > 0x7f800000U) {
        sign = 0U;
        remainder = 0U;
        return static_cast<unsigned short>(0x7fffU);
    }
    sign = x >> 31U;
    remainder = x << 16U;
    return static_cast<unsigned short>(x >> 16U);
}

__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __float2bfloat16__(const float a)
{
    __bang_bfloat16 val;
#if defined(__BANG_ARCH__)
    __asm__("cvt.gpr.rn.bf16.f32 \t %[dst], %[src];\n\t"
            : [dst] "+r"(__BFLOAT16_TO_US(val))
            : [src] "r"(a));
#else
    __bang_bfloat16_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2bfloat16(a, sign, remainder);
    if ((remainder > 0x80000000U) || ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U))) {
        r.x++;
    }
    val = r;
#endif
    return val;
}
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __float2bfloat16_rn__(const float a)
{
    __bang_bfloat16 val;
#if defined(__BANG_ARCH__)
    __asm__("cvt.gpr.rn.bf16.f32 \t %[dst], %[src];\n\t"
            : [dst] "+r"(__BFLOAT16_TO_US(val))
            : [src] "r"(a));
#else
    __bang_bfloat16_raw r;
    unsigned int sign = 0U;
    unsigned int remainder = 0U;
    r.x = __internal_float2bfloat16(a, sign, remainder);
    if ((remainder > 0x80000000U) || ((remainder == 0x80000000U) && ((r.x & 0x1U) != 0U))) {
        r.x++;
    }
    val = r;
#endif
    return val;
}

__BANG_HOSTDEVICE_BF16_DECL__ float __internal_bfloat162float(const unsigned short h)
{
    float f;
#if defined(__BANG_ARCH__)
  __asm__("cvt.gpr.rn.f32.bf16 \t %[dst], %[src];\n\t"
          : [dst] "+r"(f)
          : [src] "r"(h));
#else
    unsigned int u = static_cast<unsigned int>(h) << 16;
    f = __internal_uint_as_float(u);
#endif
    return f;
}

__BANG_HOSTDEVICE_BF16_DECL__ float __bfloat162float__(const __bang_bfloat16 a)
{
  return __internal_bfloat162float(static_cast<__bang_bfloat16_raw>(a).x);
}

__BANG_HOSTDEVICE_BF16_DECL__ int __internal_bfloat162int_rz(const __bang_bfloat16 h)
{
    float f = __bfloat162float__(h);
    int   i;
#if defined(__BANG_ARCH__)
    if (((*(int *)&f) & 0x7FFFFFFF) - 0x7F800000 > 0) {
      f = 0;
    }

    __asm__("cvt.gpr.tz.s32.f32 \t %[dst], %[src];\n\t"
            : [dst] "+r"(i)
            : [src] "r"(f));
#else
    const int max_val = (int)0x7fffffffU;
    const int min_val = (int)0x80000000U;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__bang_bfloat16_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xFF00U) {
        // NaN
        i = 0;
    } else if (f >= static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        i = static_cast<int>(f);
    }
#endif
    return i;
}

__BANG_HOSTDEVICE_BF16_DECL__ int __bfloat162int_rz__(const __bang_bfloat16 h)
{
#if defined(__BANG_ARCH__)
  int val;
  __asm__("cvt.gpr.tz.s32.bf16 \t %[dst], %[src];\n\t"
          : [dst] "+r"(val)
          : [src] "r"(__BFLOAT16_TO_CUS(h)));
  return val;
#else
    return __internal_bfloat162int_rz(h);
#endif
}

__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __int2bfloat16_rn__(const int i)
{
#if defined(__BANG_ARCH__)
  __bang_bfloat16 val;
  __asm__("cvt.gpr.rn.bf16.s32 \t %[dst], %[src];\n\t"
          : [dst] "+r"(__BFLOAT16_TO_US(val))
          : [src] "r"(i));
  return val;
#else
    const float d = static_cast<float>(i);
    return __float2bfloat16__(d);
#endif
}

__BANG_HOSTDEVICE_BF16_DECL__ short int __bfloat162short_rz__(const __bang_bfloat16 h)
{
    short int val;
#if defined(__BANG_ARCH__)
    __asm__("cvt.gpr.tz.s16.bf16 \t %[dst], %[src]; \n\t"
            : [dst] "+r"(__BFLOAT16_TO_US(val))
            : [src] "r"(__BFLOAT16_TO_CUS(h)));
#else
    const float f = __bfloat162float__(h);
    const short int max_val = (short int)0x7fffU;
    const short int min_val = (short int)0x8000U;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__bang_bfloat16_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xFF00U) {
        // NaN
        val = 0;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        val = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        val = min_val;
    } else {
        val = static_cast<short int>(f);
    }
#endif
   return val;
}

__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __short2bfloat16_rn__(const short int i)
{
#if defined(__BANG_ARCH__)
    __bang_bfloat16 val;
    __asm__("cvt.gpr.rn.bf16.s16 \t %[dst], %[src]; \n\t"
            : [dst] "+r"(__BFLOAT16_TO_US(val))
            : [src] "r"(i));
    return val;
#else
    const float f = static_cast<float>(i);
    return __float2bfloat16_rn__(f);
#endif
}

__BANG_HOSTDEVICE_BF16_DECL__ unsigned int __internal_bfloat162uint_rz(const __bang_bfloat16 h)
{
    float f = __bfloat162float__(h);
    unsigned int i;
#if defined(__BANG_ARCH__)
    if (((*(int *)&f) & 0x7FFFFFFF) - 0x7F800000 > 0) {
      f = 0;
    }

    __asm__("cvt.gpr.tz.u32.f32 \t %[dst], %[src];\n\t"
            : [dst] "+r"(i)
            : [src] "r"(f));
#else
    const unsigned int max_val = 0xffffffffU;
    const unsigned int min_val = 0U;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__bang_bfloat16_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xFF00U) {
        // NaN
        i = 0U;
    } else if (f >= static_cast<float>(max_val)) {
        // saturate maximum
        i = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        i = min_val;
    } else {
        i = static_cast<unsigned int>(f);
    }
#endif
    return i;
}

__BANG_HOSTDEVICE_BF16_DECL__ unsigned int __bfloat162uint_rz__(const __bang_bfloat16 h)
{
  return __internal_bfloat162uint_rz(h);
}

__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __uint2bfloat16_rn__(const unsigned int i)
{
#if defined(__BANG_ARCH__)
  __bang_bfloat16 val;
  __asm__("cvt.gpr.rn.bf16.u32 \t %[dst], %[src];\n\t"
          : [dst] "+r"(__BFLOAT16_TO_US(val))
          : [src] "r"(i));
  return val;
#else
    const float d = static_cast<float>(i);
    return __float2bfloat16__(d);
#endif
}

__BANG_HOSTDEVICE_BF16_DECL__ unsigned short int __bfloat162ushort_rz__(const __bang_bfloat16 h)
{
   unsigned short int val;
#if defined(__BANG_ARCH__)
  float tmp;
  __asm__("sll.gpr.u32 \t %[dst], %[src], 16; \n\t"
          : [dst] "+r"(tmp)
          : [src] "r"(__BFLOAT16_TO_CUS(h)));

  if (((*(int *)&tmp) & 0x7FFFFFFF) - 0x7F800000 > 0) {
    tmp = 0;
  }

  __asm__("cvt.gpr.tz.u16.f32 \t %[dst], %[src]; \n\t"
          : [dst] "+r"(__BFLOAT16_TO_US(val))
          : [src] "r"(tmp));
#else
    const float f = __bfloat162float__(h);
    const unsigned short int max_val = 0xffffU;
    const unsigned short int min_val = 0U;
    const unsigned short bits = static_cast<unsigned short>(static_cast<__bang_bfloat16_raw>(h).x << 1U);
    // saturation fixup
    if (bits > (unsigned short)0xFF00U) {
        // NaN
        val = 0U;
    } else if (f > static_cast<float>(max_val)) {
        // saturate maximum
        val = max_val;
    } else if (f < static_cast<float>(min_val)) {
        // saturate minimum
        val = min_val;
    } else {
        val = static_cast<unsigned short int>(f);
    }
#endif
   return val;
}

__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __ushort2bfloat16_rn__(const unsigned short int i)
{
#if defined(__BANG_ARCH__)
    __bang_bfloat16 val;
    __asm__("cvt.gpr.rn.bf16.u16 \t %[dst], %[src];\n\t"
            : [dst] "+r"(__BFLOAT16_TO_US(val))
            : [src] "r"(i));
    return val;
#else
    const float f = static_cast<float>(i);
    return __float2bfloat16_rn__(f);
#endif
}

__BANG_HOSTDEVICE_BF16_DECL__ bool __hisnan(const __bang_bfloat16 a) {
    const __bang_bfloat16_raw hr = static_cast<__bang_bfloat16_raw>(a);
    return ((hr.x & 0x7FFFU) > 0x7F80U);
}

__BANG_HOSTDEVICE_BF16_DECL__ long long int __bfloat162ll_rz(const __bang_bfloat16 h)
{
  long long int i;
#if defined(__BANG_ARCH__)
  __asm__ volatile("cvt.gpr.tz.s64.bf16 \t %[dst], %[src];\n\t"
                   : [dst] "=l"(i)
                   : [src] "r"(__BFLOAT16_TO_CUS(h)));
  return i;
#else
  const float f = __bfloat162float__(h);
  const long long int max_val = (long long int)0x7fffffffffffffffULL;
  const long long int min_val = (long long int)0x8000000000000000ULL;
  const unsigned short bits = static_cast<unsigned short>(static_cast<__bang_bfloat16_raw>(h).x << 1U);
  // saturation fixup
  if (bits > (unsigned short)0xFF00U) {
    // NaN
    i = min_val;
  } else if (f >= static_cast<float>(max_val)) {
    // saturate maximum
    i = max_val;
  } else if (f < static_cast<float>(min_val)) {
    // saturate minimum
    i = min_val;
  } else {
    i = static_cast<long long int>(f);
  }
  return i;
#endif
}

__BANG_HOSTDEVICE_BF16_DECL__ unsigned long long int __bfloat162ull_rz(const __bang_bfloat16 h)
{
  unsigned long long int i;
#if !defined(__BANG_ARCH__) || (__BANG_ARCH__ == 592) || (__BANG_ARCH__ == 506) || (__BANG_ARCH__ == 606) || (__BANG_ARCH__ == 607)
  const float f = __bfloat162float__(h);
  const unsigned long long int max_val = 0xffffffffffffffffULL;
  const unsigned long long int min_val = 0ULL;
  const unsigned short bits = static_cast<unsigned short>(static_cast<__bang_bfloat16_raw>(h).x << 1U);
  // saturation fixup
  if (bits > (unsigned short)0xFF00U) {
    // NaN
    i = 0x8000000000000000ULL;
  } else if (f >= static_cast<float>(max_val)) {
    // saturate maximum
    i = max_val;
  } else if (f < static_cast<float>(min_val)) {
    // saturate minimum
    i = min_val;
  } else {
    i = static_cast<unsigned long long int>(f);
  }
  return i;
#else
   __asm__ volatile("cvt.gpr.tz.u64.bf16 \t %[dst], %[src];\n\t"
                    : [dst] "=l"(i)
                    : [src] "r"(__BFLOAT16_TO_CUS(h)));
   return i;
#endif
}

__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __ll2bfloat16_rn(const long long int i)
{
#if defined(__BANG_ARCH__)
  __bang_bfloat16 h;
  __asm__ volatile("cvt.gpr.rn.bf16.s64 \t %[dst], %[src];\n\t"
                   : [dst] "+r"(__BFLOAT16_TO_US(h))
                   : [src] "l"(i));
  return h;
#else
  float f = static_cast<float>(i);
  const long long int lf = static_cast<long long int>(f);
  unsigned int u = __internal_float_as_uint(f);

  if ((f > 0.0f) && (lf > i)) {
    u--;
  }
  if ((f < 0.0f) && (lf < i)) {
    u--;
  }
  if (lf != i) {
    u |= 1U;
  }

  f = __internal_uint_as_float(u);
  return __float2bfloat16_rn__(f);
#endif
}

__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __ull2bfloat16_rn(const unsigned long long int i)
{
#if defined(__BANG_ARCH__)
  __bang_bfloat16 h;
  __asm__ volatile("cvt.gpr.rn.bf16.u64 \t %[dst], %[src];\n\t"
                   : [dst] "+r"(__BFLOAT16_TO_US(h))
                   : [src] "l"(i));
  return h;
#else
  float f = static_cast<float>(i);
  const unsigned long long int uf = static_cast<unsigned long long int>(f);
  unsigned int u = __internal_float_as_uint(f);
  // round up happened here
  // note: no need to handle round up to f == 0x1.p64 specially
  if (uf > i) {
    u--;
  }
  if (uf != i) {
    u |= 1U;
  }
  f = __internal_uint_as_float(u);
  return __float2bfloat16_rn__(f);
#endif
}

/******************************************************************************
*                             __bang_bfloat16 comparison                              *
******************************************************************************/
__BANG_HOSTDEVICE_BF16_DECL__ bool __heq(const __bang_bfloat16 a, const __bang_bfloat16 b) {
    const float fa = __bfloat162float__(a);
    const float fb = __bfloat162float__(b);
    return (fa == fb);
}
__BANG_HOSTDEVICE_BF16_DECL__ bool __hne(const __bang_bfloat16 a, const __bang_bfloat16 b) {
    const float fa = __bfloat162float__(a);
    const float fb = __bfloat162float__(b);
    return (fa != fb) && (!__hisnan(a) && (!__hisnan(b)));
}
__BANG_HOSTDEVICE_BF16_DECL__ bool __hle(const __bang_bfloat16 a, const __bang_bfloat16 b) {
    float fa = __bfloat162float__(a);
    float fb = __bfloat162float__(b);
    return (fa <= fb);
}
__BANG_HOSTDEVICE_BF16_DECL__ bool __hge(const __bang_bfloat16 a, const __bang_bfloat16 b) {
    float fa = __bfloat162float__(a);
    float fb = __bfloat162float__(b);
    return (fa >= fb);
}
__BANG_HOSTDEVICE_BF16_DECL__ bool __hlt(const __bang_bfloat16 a, const __bang_bfloat16 b) {
    float fa = __bfloat162float__(a);
    float fb = __bfloat162float__(b);
    return (fa < fb);
}
__BANG_HOSTDEVICE_BF16_DECL__ bool __hgt(const __bang_bfloat16 a, const __bang_bfloat16 b) {
    float fa = __bfloat162float__(a);
    float fb = __bfloat162float__(b);
    return (fa > fb);
}
__BANG_HOSTDEVICE_BF16_DECL__ bool __hequ(const __bang_bfloat16 a, const __bang_bfloat16 b) {
    float fa = __bfloat162float__(a);
    float fb = __bfloat162float__(b);
    return (fa == fb) || (__hisnan(a)) || (__hisnan(b));
}
__BANG_HOSTDEVICE_BF16_DECL__ bool __hneu(const __bang_bfloat16 a, const __bang_bfloat16 b) {
    float fa = __bfloat162float__(a);
    float fb = __bfloat162float__(b);
    return (fa != fb);
}
__BANG_HOSTDEVICE_BF16_DECL__ bool __hleu(const __bang_bfloat16 a, const __bang_bfloat16 b) {
    float fa = __bfloat162float__(a);
    float fb = __bfloat162float__(b);
    return (fa <= fb) || (__hisnan(a)) || (__hisnan(b));
}
__BANG_HOSTDEVICE_BF16_DECL__ bool __hgeu(const __bang_bfloat16 a, const __bang_bfloat16 b) {
    float fa = __bfloat162float__(a);
    float fb = __bfloat162float__(b);
    return (fa >= fb) || (__hisnan(a)) || (__hisnan(b));
}
__BANG_HOSTDEVICE_BF16_DECL__ bool __hltu(const __bang_bfloat16 a, const __bang_bfloat16 b) {
    float fa = __bfloat162float__(a);
    float fb = __bfloat162float__(b);
    return (fa < fb) || (__hisnan(a)) || (__hisnan(b));
}
__BANG_HOSTDEVICE_BF16_DECL__ bool __hgtu(const __bang_bfloat16 a, const __bang_bfloat16 b) {
    float fa = __bfloat162float__(a);
    float fb = __bfloat162float__(b);
    return (fa > fb) || (__hisnan(a)) || (__hisnan(b));
}
#undef __COMPARISON_OP_BFLOAT16_MACRO
/******************************************************************************
*                             __bang_bfloat16 arithmetic                      *
******************************************************************************/
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __hadd(const __bang_bfloat16 a, const __bang_bfloat16 b) {
  __bang_bfloat16 val;
  float fa = __bfloat162float__(a);
  float fb = __bfloat162float__(b);
  float fc = fa + fb;
  val = __float2bfloat16__(fc);
  return val;
}
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __hsub(const __bang_bfloat16 a, const __bang_bfloat16 b) {
  __bang_bfloat16 val;
  float fa = __bfloat162float__(a);
  float fb = __bfloat162float__(b);
  float fc = fa - fb;
  val = __float2bfloat16__(fc);
  return val;
}
__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __hmul(const __bang_bfloat16 a, const __bang_bfloat16 b)
{
  __bang_bfloat16 val;
  float fa = __bfloat162float__(a);
  float fb = __bfloat162float__(b);
  float fc = fa * fb;
  val = __float2bfloat16__(fc);
  return val;
}

__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __hdiv(const __bang_bfloat16 a, const __bang_bfloat16 b) {
  __bang_bfloat16 val;
  float fa = __bfloat162float__(a);
  float fb = __bfloat162float__(b);
  float fc = fa / fb;
  val = __float2bfloat16__(fc);
  return val;
}

__BANG_HOSTDEVICE_BF16_DECL__ __bang_bfloat16 __hneg(const __bang_bfloat16 a) {
  float fa = __bfloat162float__(a);
  return __float2bfloat16__(-fa);
}
#endif /* defined(__cplusplus) */

#undef __BANG_HOSTDEVICE_BF16_DECL__

#if defined(__cplusplus) && !defined(BANG_NO_BFLOAT16)
typedef __bang_bfloat16  bfloat16_t;
typedef __bang_bfloat16_raw bfloat16_raw_t;
#endif /* defined(__cplusplus) && !defined(BANG_NO_BFLOAT16) */

#if defined(__CPP_VERSION_AT_LEAST_11_BF16)
#undef __CPP_VERSION_AT_LEAST_11_BF16
#endif /* defined(__CPP_VERSION_AT_LEAST_11_BF16) */

#endif /* end of include guard: __BANG_BF16_H__ */
