#pragma once
struct acFloatComplex { float x, y; };
struct acDoubleComplex { double x, y; };
static inline acFloatComplex  make_acFloatComplex(float r, float i){ return {r,i}; }
static inline acDoubleComplex make_acDoubleComplex(double r, double i){ return {r,i}; }
static inline float  acCrealf(acFloatComplex z){ return z.x; }
static inline float  acCimagf(acFloatComplex z){ return z.y; }
static inline double acCreal(acDoubleComplex z){ return z.x; }
static inline double acCimag(acDoubleComplex z){ return z.y; }
