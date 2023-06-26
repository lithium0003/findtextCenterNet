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

#include "custom_fill.h"
#include "tensorflow/core/util/gpu_kernel_helper.h"

namespace tensorflow {
namespace functor {

typedef Eigen::GpuDevice GPUDevice;

__device__ int find_startidx_d(const int *sortidx, const float *position, int count, float xi) 
{
  int lower_idx = 0;
  int upper_idx = count;
  while(lower_idx < upper_idx) {
    int current_idx = (lower_idx + upper_idx) / 2;
    if(current_idx == lower_idx || current_idx == upper_idx) break;

    float cx = position[sortidx[current_idx]*4 + 0] / 2;
    float w = position[sortidx[current_idx]*4 + 2] / 2;

    if(cx + w < xi) {
      lower_idx = current_idx;
    }
    else{
      upper_idx = current_idx;
    }
  }
  return lower_idx;
}

// Define the CUDA kernel.
template <typename T>
__global__ void CustomFillACudaKernel(
    const float *position, const int32_t *code_list, int count,
    const int *sortidx,
    int width, int height,
    T* out) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x < width && y < height) {
    float xi = x - (width - 1) / 2.0;
    float yi = y - (height - 1) / 2.0;

    out[(y * width + x)*2 + 0] = 0;
    out[(y * width + x)*2 + 1] = 0;

    for(int i = find_startidx_d(sortidx, position, count, xi); i < count; i++) {
      float cx = position[sortidx[i]*4 + 0] / 2;
      float cy = position[sortidx[i]*4 + 1] / 2;
      float w = position[sortidx[i]*4 + 2] / 2;
      float h = position[sortidx[i]*4 + 3] / 2;

      float w2 = max(w / 2, 2.0f);
      float h2 = max(h / 2, 2.0f);

      if(cx - w2 < xi && xi < cx + w2 && cy - h2 < yi && yi < cy + h2) {
        float x2 = (xi - cx) / w2;
        float y2 = (yi - cy) / h2;
        if(x2 * x2 + y2 * y2 < 1) {
          int32_t code = code_list[sortidx[i]*2 + 0];
          int32_t opcode = code_list[sortidx[i]*2 + 1];

          out[(y * width + x)*2 + 0] = code;
          out[(y * width + x)*2 + 1] = opcode;
          break;
        }
      }
    }
  }
}

template <typename T>
__global__ void CustomFillBCudaKernel(
    const float *position, const float *angle, int count,
    const int *sortidx,
    int width, int height,
    T* out) {

  int x = blockIdx.x * blockDim.x + threadIdx.x;
  int y = blockIdx.y * blockDim.y + threadIdx.y;
  if(x < width && y < height) {
    float xi = x - (width - 1) / 2.0;
    float yi = y - (height - 1) / 2.0;

    out[(y * width + x)*4 + 0] = 0;
    out[(y * width + x)*4 + 1] = 0;
    out[(y * width + x)*4 + 2] = 0;
    out[(y * width + x)*4 + 3] = 0;

    for(int i = find_startidx_d(sortidx, position, count, xi); i < count; i++) {
      float cx = position[sortidx[i]*4 + 0] / 2;
      float cy = position[sortidx[i]*4 + 1] / 2;
      float w = position[sortidx[i]*4 + 2] / 2;
      float h = position[sortidx[i]*4 + 3] / 2;

      float w2 = max(w / 2, 2.0f);
      float h2 = max(h / 2, 2.0f);

      if(cx - w2 < xi && xi < cx + w2 && cy - h2 < yi && yi < cy + h2) {
        float x0 = xi - cx;
        float y0 = yi - cy;
        float x2 = x0 / w2;
        float y2 = y0 / h2;
        if(x2 * x2 + y2 * y2 < 1) {
          float fixw = w * fabs(cos(*angle)) + h * fabs(sin(*angle));
          float fixh = h * fabs(cos(*angle)) + w * fabs(sin(*angle));
          fixw = log(fixw / 512) + 3;
          fixh = log(fixh / 512) + 3;
          float offset_x = -(x0 * cos(*angle) + y0 * sin(*angle));
          float offset_y = -(y0 * sin(*angle + M_PI / 2) + x0 * cos(*angle + M_PI / 2));

          out[(y * width + x)*4 + 0] = fixw;
          out[(y * width + x)*4 + 1] = fixh;
          out[(y * width + x)*4 + 2] = offset_x;
          out[(y * width + x)*4 + 3] = offset_y;
          break;
        }
      }
    }
  }
}

template <typename T>
__global__ void CustomFillCCudaKernel(
    const float *position, int count,
    int width, int height,
    T* out) {

  int id = blockIdx.x * blockDim.x + threadIdx.x;

  if (id < count) {
    float cx = position[id*4 + 0] / 2;
    float cy = position[id*4 + 1] / 2;
    float w = position[id*4 + 2] / 2;
    float h = position[id*4 + 3] / 2;

    if(fabs(cx) > width/2.0 || fabs(cy) > height/2.0) {
      return;
    }

    float fix_w = max(w / 2, 4.0f);
    float fix_h = max(h / 2, 4.0f);
    int kernel_size = max(fix_w, fix_h);
    float std_x = fix_w / 4;
    float std_y = fix_h / 4;

    int x1 = max(0.0, cx - w/2 + (width - 1) / 2.0);
    int x2 = min(width * 1.0 , cx + w/2 + (width - 1) / 2.0);
    int y1 = max(0.0, cy - h/2 + (height - 1) / 2.0);
    int y2 = min(height * 1.0, cy + h/2 + (height - 1) / 2.0);

    for(int y = y1; y < y2; y++) {
      for(int x = x1; x < x2; x++) {
        float xi = x - (width - 1) / 2.0;
        float yi = y - (height - 1) / 2.0;

        int x3 = xi - cx;
        int y3 = yi - cy;
        if(x3 > -kernel_size && x3 < kernel_size && y3 > -kernel_size && y3 < kernel_size) {
          float gaussx = expf(-0.5 * x3 * x3 / (std_x * std_x));
          float gaussy = expf(-0.5 * y3 * y3 / (std_y * std_y));

          out[y * width + x] = max(out[y * width + x], gaussx * gaussy);
        }
      }
    }
  }
}

// Define the GPU implementation that launches the CUDA kernel.
template <typename T>
struct CustomFillAFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d,
    const float *position, const int32_t *code_list, int count,
    const int *sortidx,
    int width, int height,
    T* out) {

    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    dim3 block_count(width / 32 + 1, height / 32 + 1, 1);
    dim3 thread_per_block(32, 32, 1);
    CustomFillACudaKernel<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(position, code_list, count, 
        sortidx,
        width, height, out);
  }
};

template <typename T>
struct CustomFillBFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d,
    const float *position, const float *angle, int count,
    const int *sortidx,
    int width, int height,
    T* out) {

    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.
    dim3 block_count(width / 32 + 1, height / 32 + 1, 1);
    dim3 thread_per_block(32, 32, 1);
    CustomFillBCudaKernel<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(position, angle, count, 
        sortidx,
        width, height, out);
  }
};

template <typename T>
struct CustomFillCFunctor<GPUDevice, T> {
  void operator()(const GPUDevice& d,
    const float *position, int count,
    int width, int height,
    T* out) {

    // Launch the cuda kernel.
    //
    // See core/util/cuda_kernel_helper.h for example of computing
    // block count and thread_per_block count.

    cudaMemsetAsync(out, 0, sizeof(T)*width*height, d.stream());

    dim3 block_count(count / 1024 + 1, 1, 1);
    dim3 thread_per_block(1024, 1, 1);
    CustomFillCCudaKernel<T>
        <<<block_count, thread_per_block, 0, d.stream()>>>(position, count, 
        width, height, out);
  }
};

// Explicitly instantiate functors for the types of OpKernels registered.
template struct CustomFillAFunctor<GPUDevice, int32>;
template struct CustomFillAFunctor<GPUDevice, int64>;

template struct CustomFillBFunctor<GPUDevice, float>;

template struct CustomFillCFunctor<GPUDevice, float>;

}  // end namespace functor
}  // end namespace tensorflow

#endif  // GOOGLE_CUDA
