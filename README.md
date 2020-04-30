# cudart-sang-rhie
# Basic Ray Tracing in Cuda
Skylar Sang and Matthew Rhie
RPI ECSE 4740 Spring 2020

# Resources used
This project was adapted by the book  *Ray Tracing in One Weekend* by Peter Shirley which was used to learn ray tracing fundamentals and mathematics.

Book [Here](https://raytracing.github.io/books/RayTracingInOneWeekend.html)

Guidance for Parallelization with CUDA and other library suggestions by Nvidia Developer Roger Allen

[Blog Post]
(https://devblogs.nvidia.com/accelerated-ray-tracing-cuda/)

Because of the emphasis on parallelizing the ray tracer as opposed to building the ray tracer itself for this project, the mathematical concepts implementations of the ray tracer are primarily derivative of how they were taught in the book and blog post. 

----
## Program Build/Run

Program uses standard CUDA C++ compilation using NVCC:

>nvcc -O3 -gencode arch=compute\_70,code=sm\_70 main.cu -o
rt

and to run:
> ./rt 256 > output.ppm

where 256 indicates the number of threads per block used to render the image in the kernel. The ppm file can be converted to an image binary using an online ppm viewer or a module such as ppmtojpeg or similar.

