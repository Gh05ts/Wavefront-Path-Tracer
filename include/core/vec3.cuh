#pragma once

#include <cuda_runtime.h>
#include <cmath>

struct Vec3 {
    float x;
    float y;
    float z;

    __host__ __device__
    Vec3(): x(0.0f), y(0.0f), z(0.0f) {}

    __host__ __device__
    Vec3(float x, float y, float z): x(x), y(y), z(z) {}

    __host__ __device__
    Vec3 operator-() const {
        return Vec3(-x, -y, -z);
    }

    __host__ __device__
    Vec3 operator+(const Vec3& v) const {
        return Vec3(x + v.x, y + v.y, z + v.z);
    }

    __host__ __device__
    Vec3 operator-(const Vec3& v) const {
        return Vec3(x - v.x, y - v.y, z - v.z);
    }

    __host__ __device__
    Vec3 operator*(float s) const {
        return Vec3(x * s, y * s, z * s);
    }

    __host__ __device__
    Vec3 operator/(float s) const {
        float inv = 1.0f / s;
        return *this * inv;
    }

    __host__ __device__
    Vec3& operator+=(const Vec3& v) {
        x += v.x;
        y += v.y;
        z += v.z;
        return *this;
    }

    __host__ __device__
    Vec3& operator*=(float s) {
        x *= s;
        y *= s;
        z *= s;
        return *this;
    }
};

__host__ __device__
inline Vec3 operator*(float s, const Vec3& v) {
    return v * s;
}

__host__ __device__
inline float dot(const Vec3& a, const Vec3& b) {
    return a.x * b.x + a.y * b.y + a.z * b.z;
}

__host__ __device__
inline Vec3 cross(const Vec3& a, const Vec3& b) {
    return Vec3(a.y * b.z - a.z * b.y, a.z * b.x - a.x * b.z, a.x * b.y - a.y * b.x);
}

__host__ __device__
inline float lengthSquared(const Vec3& v) {
    return dot(v, v);
}

__host__ __device__
inline float length(const Vec3& v) {
    return sqrtf(lengthSquared(v));
}

__host__ __device__
inline Vec3 normalize(const Vec3& v) {
    float invLength = 1.0f / sqrtf(dot(v, v));
    return v * invLength;
}

__host__ __device__
inline Vec3 hadamard(const Vec3& a, const Vec3& b) {
    return Vec3(a.x * b.x, a.y * b.y, a.z * b.z);
}
