/* 
Skylar Sang & Matthew Rhie
ECSE 4740
Spring 2020

Supplmental File adapted from the book Ray Tracing in One Weekend by Peter Shirley
*/

#ifndef HITABLEH
#define HITABLEH

#include "ray.h"

class material;

struct hit_record
{
    float t;
    vec3 p;
    vec3 normal;
    material *mat_ptr;
};

class hitable  {
    public:
        __device__ virtual bool hit(const ray& r, float t_min, float t_max, hit_record& rec) const = 0;
};

#endif
