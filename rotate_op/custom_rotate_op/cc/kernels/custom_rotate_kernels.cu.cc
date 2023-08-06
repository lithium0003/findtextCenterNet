/* Copyright 2019 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#if GOOGLE_CUDA

#define EIGEN_USE_GPU

#include "custom_rotate.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

// Define the CUDA kernel.
template <typename T>
__global__ void CustomRotate1CudaKernel(
    int im_width, int im_height, int im_ch, 
    int width, int height,
    const float *rot_cx, const float *rot_cy,
    const float *sx, const float *sy, const float *angle,
    const T* in, float* out) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x < width && y < height) {
    float im_x = *sx * ((x - (width - 1) / 2.0) * cos(*angle) - (y - (height - 1) / 2.0) * sin(*angle)) + *rot_cx;
    float im_y = *sy * ((x - (width - 1) / 2.0) * sin(*angle) + (y - (height - 1) / 2.0) * cos(*angle)) + *rot_cy;
    int im_xi = im_x;
    int im_yi = im_y;
    if(im_xi < 0 || im_xi >= im_width || im_yi < 0 || im_yi >= im_height) {
      for(int c = 0; c < im_ch; c++) {
        out[(y * width + x)*im_ch + c] = 0;
      }
    }
    else {
      for(int c = 0; c < im_ch; c++) {
        float p11 = in[(im_yi * im_width + im_xi)*im_ch + c];
        float p12 = p11;
        float p21 = p11;
        float p22 = p11;
        if(im_xi + 1 < im_width) {
          p12 = in[(im_yi * im_width + im_xi + 1)*im_ch + c];
        }
        if(im_yi + 1 < im_height) {
          p21 = in[((im_yi + 1) * im_width + im_xi)*im_ch + c];
        }
        if(im_xi + 1 < im_width && im_yi + 1 < im_height) {
          p22 = in[((im_yi + 1) * im_width + im_xi + 1)*im_ch + c];
        }
        float dx = im_x - im_xi;
        float dy = im_y - im_yi;
        out[(y * width + x)*im_ch + c] = (1 - dx) * (1 - dy) * p11 + dx * (1 - dy) * p12 + (1 - dx) * dy * p21 + dx * dy * p22;
      }
    }
  }
}

template <typename T>
__global__ void CustomRotate2CudaKernel(
    int im_width, int im_height, int im_ch, 
    int width, int height,
    const float *rot_cx, const float *rot_cy,
    const float *sx, const float *sy, const float *angle,
    const T* in, T* out) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x < width && y < height) {
    float im_x = *sx * ((x - (width - 1) / 2.0) * cos(*angle) - (y - (height - 1) / 2.0) * sin(*angle)) + *rot_cx;
    float im_y = *sy * ((x - (width - 1) / 2.0) * sin(*angle) + (y - (height - 1) / 2.0) * cos(*angle)) + *rot_cy;
    int im_xi = im_x;
    int im_yi = im_y;
    if(im_xi < 0 || im_xi >= im_width || im_yi < 0 || im_yi >= im_height) {
      for(int c = 0; c < im_ch; c++) {
        out[(y * width + x)*im_ch + c] = 0;
      }
    }
    else {
      for(int c = 0; c < im_ch; c++) {
        T p11 = in[(im_yi * im_width + im_xi)*im_ch + c];
        T p12 = p11;
        T p21 = p11;
        T p22 = p11;
        if(im_xi + 1 < im_width) {
          p12 = in[(im_yi * im_width + im_xi + 1)*im_ch + c];
        }
        if(im_yi + 1 < im_height) {
          p21 = in[((im_yi + 1) * im_width + im_xi)*im_ch + c];
        }
        if(im_xi + 1 < im_width && im_yi + 1 < im_height) {
          p22 = in[((im_yi + 1) * im_width + im_xi + 1)*im_ch + c];
        }
        float dx = im_x - im_xi;
        float dy = im_y - im_yi;
        if(dx < 0.5) {
          if(dy < 0.5) {
            out[(y * width + x)*im_ch + c] = p11;    
          }
          else {
            out[(y * width + x)*im_ch + c] = p21;    
          }
        }
        else {
          if(dy < 0.5) {
            out[(y * width + x)*im_ch + c] = p12;    
          }
          else {
            out[(y * width + x)*im_ch + c] = p22;    
          }
        }
      }
    }
  }
}

template <typename T>
__global__ void CustomRotate3CudaKernel(
    int im_width, int im_height, int im_ch, 
    int width, int height,
    const float *rot_cx, const float *rot_cy,
    const float *sx, const float *sy, const float *angle,
    const T* in, float* out) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x < width && y < height) {
    float im_x = *sx * ((x - (width - 1) / 2.0) * cos(*angle) - (y - (height - 1) / 2.0) * sin(*angle)) + *rot_cx;
    float im_y = *sy * ((x - (width - 1) / 2.0) * sin(*angle) + (y - (height - 1) / 2.0) * cos(*angle)) + *rot_cy;
    int im_xi = im_x;
    int im_yi = im_y;
    if(im_xi < 0 || im_xi >= im_width || im_yi < 0 || im_yi >= im_height) {
      for(int c = 0; c < im_ch; c++) {
        out[(y * width + x)*im_ch + c] = 0;
      }
    }
    else {
      for(int c = 0; c < im_ch; c++) {
        float p11 = in[(im_yi * im_width + im_xi)*im_ch + c];
        float p12 = p11;
        float p21 = p11;
        float p22 = p11;
        if(im_xi + 1 < im_width) {
          p12 = in[(im_yi * im_width + im_xi + 1)*im_ch + c];
        }
        if(im_yi + 1 < im_height) {
          p21 = in[((im_yi + 1) * im_width + im_xi)*im_ch + c];
        }
        if(im_xi + 1 < im_width && im_yi + 1 < im_height) {
          p22 = in[((im_yi + 1) * im_width + im_xi + 1)*im_ch + c];
        }
        if(p11 >= 1.0 || p12 >= 1.0 || p21 >= 1.0 || p22 >= 1.0) {
          out[(y * width + x)*im_ch + c] = 1.0;  
        }
        else {
          float dx = im_x - im_xi;
          float dy = im_y - im_yi;
          out[(y * width + x)*im_ch + c] = (1 - dx) * (1 - dy) * p11 + dx * (1 - dy) * p12 + (1 - dx) * dy * p21 + dx * dy * p22;
        }
      }
    }
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct CustomRotate1Functor<GPUDevice, T> {
  void operator()(const GPUDevice& d,
    int im_width, int im_height, int im_ch, 
    int width, int height,
    const float *rot_cx, const float *rot_cy,
    const float *sx, const float *sy, const float *angle,
    const T* in, float* out) {

    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    dim3 block_count(width / 32 + 1, height / 32 + 1, 1);
    dim3 thread_per_block(32, 32, 1);
    CustomRotate1CudaKernel<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(im_width, im_height, im_ch, 
        width, height, rot_cx, rot_cy, sx, sy, angle, in, out);
  }
};

template <typename T>
struct CustomRotate2Functor<GPUDevice, T> {
  void operator()(const GPUDevice& d,
    int im_width, int im_height, int im_ch, 
    int width, int height,
    const float *rot_cx, const float *rot_cy,
    const float *sx, const float *sy, const float *angle,
    const T* in, T* out) {

    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    dim3 block_count(width / 32 + 1, height / 32 + 1, 1);
    dim3 thread_per_block(32, 32, 1);
    CustomRotate2CudaKernel<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(im_width, im_height, im_ch, 
        width, height, rot_cx, rot_cy, sx, sy, angle, in, out);
  }
};

template <typename T>
struct CustomRotate3Functor<GPUDevice, T> {
  void operator()(const GPUDevice& d,
    int im_width, int im_height, int im_ch, 
    int width, int height,
    const float *rot_cx, const float *rot_cy,
    const float *sx, const float *sy, const float *angle,
    const T* in, float* out) {

    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    dim3 block_count(width / 32 + 1, height / 32 + 1, 1);
    dim3 thread_per_block(32, 32, 1);
    CustomRotate3CudaKernel<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(im_width, im_height, im_ch, 
        width, height, rot_cx, rot_cy, sx, sy, angle, in, out);
  }
};


// Explicitly instantiate functors for the types of OpKernels registered.
template struct CustomRotate1Functor<GPUDevice, float>;
template struct CustomRotate1Functor<GPUDevice, double>;
template struct CustomRotate1Functor<GPUDevice, int32>;
template struct CustomRotate1Functor<GPUDevice, int64>;
template struct CustomRotate1Functor<GPUDevice, uint8>;
template struct CustomRotate1Functor<GPUDevice, int16>;
template struct CustomRotate1Functor<GPUDevice, int8>;
template struct CustomRotate1Functor<GPUDevice, uint16>;
template struct CustomRotate1Functor<GPUDevice, uint32>;
template struct CustomRotate1Functor<GPUDevice, uint64>;

template struct CustomRotate2Functor<GPUDevice, float>;
template struct CustomRotate2Functor<GPUDevice, double>;
template struct CustomRotate2Functor<GPUDevice, int32>;
template struct CustomRotate2Functor<GPUDevice, int64>;
template struct CustomRotate2Functor<GPUDevice, uint8>;
template struct CustomRotate2Functor<GPUDevice, int16>;
template struct CustomRotate2Functor<GPUDevice, int8>;
template struct CustomRotate2Functor<GPUDevice, uint16>;
template struct CustomRotate2Functor<GPUDevice, uint32>;
template struct CustomRotate2Functor<GPUDevice, uint64>;

template struct CustomRotate3Functor<GPUDevice, float>;
template struct CustomRotate3Functor<GPUDevice, double>;
template struct CustomRotate3Functor<GPUDevice, int32>;
template struct CustomRotate3Functor<GPUDevice, int64>;
template struct CustomRotate3Functor<GPUDevice, uint8>;
template struct CustomRotate3Functor<GPUDevice, int16>;
template struct CustomRotate3Functor<GPUDevice, int8>;
template struct CustomRotate3Functor<GPUDevice, uint16>;
template struct CustomRotate3Functor<GPUDevice, uint32>;
template struct CustomRotate3Functor<GPUDevice, uint64>;

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
