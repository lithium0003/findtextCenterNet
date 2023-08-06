/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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
#endif  // GOOGLE_CUDA

#include "custom_fill.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {

int find_startidx(const int *sortidx, const float *position, int count, float xi) 
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

// CPU specialization of actual computation.
template <typename T>
struct CustomFillAFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d,
    const float *position, const int32_t *code_list, int count,
    const int *sortidx,
    int width, int height,
    T* out) {

      for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
          float xi = x - (width - 1) / 2.0;
          float yi = y - (height - 1) / 2.0;

          out[(y * width + x)*2 + 0] = 0;
          out[(y * width + x)*2 + 1] = 0;

          for(int i = find_startidx(sortidx, position, count, xi); i < count; i++) {
            float cx = position[sortidx[i]*4 + 0] / 2;
            float cy = position[sortidx[i]*4 + 1] / 2;
            float w = position[sortidx[i]*4 + 2] / 2;
            float h = position[sortidx[i]*4 + 3] / 2;

            float w2 = std::max(w / 2, 2.0f);
            float h2 = std::max(h / 2, 2.0f);

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
  }
};

template <typename T>
struct CustomFillBFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d,
    const float *position, const float *angle, int count,
    const int *sortidx,
    int width, int height,
    T* out) {

      for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
          float xi = x - (width - 1) / 2.0;
          float yi = y - (height - 1) / 2.0;

          out[(y * width + x)*4 + 0] = 0;
          out[(y * width + x)*4 + 1] = 0;
          out[(y * width + x)*4 + 2] = 0;
          out[(y * width + x)*4 + 3] = 0;

          for(int i = find_startidx(sortidx, position, count, xi); i < count; i++) {
            float cx = position[sortidx[i]*4 + 0] / 2;
            float cy = position[sortidx[i]*4 + 1] / 2;
            float w = position[sortidx[i]*4 + 2] / 2;
            float h = position[sortidx[i]*4 + 3] / 2;

            float w2 = std::max(w / 2, 2.0f);
            float h2 = std::max(h / 2, 2.0f);

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
  }
};


template <typename T>
struct CustomFillCFunctor<CPUDevice, T> {
  void operator()(const CPUDevice& d,
    const float *position, int count,
    int width, int height,
    T* out) {

      for(int i = 0; i < width*height; i++) {
        out[i] = 0;
      }

      for(int id = 0; id < count; id++) {
        float cx = position[id*4 + 0] / 2;
        float cy = position[id*4 + 1] / 2;
        float w = position[id*4 + 2] / 2;
        float h = position[id*4 + 3] / 2;

        if(fabs(cx) > width/2.0 || fabs(cy) > height/2.0) {
          continue;
        }

        float fix_w = std::max(w / 2, 4.0f);
        float fix_h = std::max(h / 2, 4.0f);
        int kernel_size = std::max(fix_w, fix_h);
        float std_x = fix_w / 4;
        float std_y = fix_h / 4;

        int x1 = std::max(0.0, cx - w/2 + (width - 1) / 2.0);
        int x2 = std::min(width * 1.0, cx + w/2 + (width - 1) / 2.0);
        int y1 = std::max(0.0, cy - h/2 + (height - 1) / 2.0);
        int y2 = std::min(height * 1.0, cy + h/2 + (height - 1) / 2.0);

        for(int y = y1; y < y2; y++) {
          for(int x = x1; x < x2; x++) {
            float xi = x - (width - 1) / 2.0;
            float yi = y - (height - 1) / 2.0;

            int x3 = xi - cx;
            int y3 = yi - cy;
            if(x3 > -kernel_size && x3 < kernel_size && y3 > -kernel_size && y3 < kernel_size) {
              float gaussx = exp(-0.5 * x3 * x3 / (std_x * std_x));
              float gaussy = exp(-0.5 * y3 * y3 / (std_y * std_y));

              out[y * width + x] = std::max(out[y * width + x], gaussx * gaussy);
            }
          }
        }
      }
  }
};

// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class CustomFillAOp : public OpKernel {
 public:
  explicit CustomFillAOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& position_tensor = context->input(1);
    const Tensor& code_list_tensor = context->input(2);
    const Tensor& sortidx_tensor = context->input(3);

    OP_REQUIRES(context, input_tensor.shape().dims() == 3,
                errors::InvalidArgument("input image must be 3-D",
                    input_tensor.shape().DebugString()));
    OP_REQUIRES(context, position_tensor.shape().dims() == 2,
                errors::InvalidArgument("position must be 2-D",
                    position_tensor.shape().DebugString()));
    OP_REQUIRES(context, code_list_tensor.shape().dims() == 2,
                errors::InvalidArgument("code_list must be 2-D",
                    code_list_tensor.shape().DebugString()));
    OP_REQUIRES(context, sortidx_tensor.shape().dims() == 1,
                errors::InvalidArgument("sortidx must be 1-D",
                    sortidx_tensor.shape().DebugString()));
    OP_REQUIRES(context, input_tensor.shape().dim_size(2) == 2,
                errors::InvalidArgument("input channel must have 2 channel",
                    input_tensor.shape().DebugString()));
    OP_REQUIRES(context, position_tensor.shape().dim_size(1) == 4,
                errors::InvalidArgument("position must have a shape n * 4",
                    position_tensor.shape().DebugString()));
    OP_REQUIRES(context, code_list_tensor.shape().dim_size(1) == 2,
                errors::InvalidArgument("code_list must have a shape n * 2",
                    code_list_tensor.shape().DebugString()));

    int count = position_tensor.shape().dim_size(0);
    OP_REQUIRES(context, count == code_list_tensor.shape().dim_size(0),
                errors::InvalidArgument("Number elements is deferent.",
                  position_tensor.shape().DebugString(),
                  code_list_tensor.shape().DebugString()));
    OP_REQUIRES(context, count == sortidx_tensor.shape().dim_size(0),
                errors::InvalidArgument("Number elements is deferent.",
                  position_tensor.shape().DebugString(),
                  sortidx_tensor.shape().DebugString()));

    int height = input_tensor.shape().dim_size(0);
    int width = input_tensor.shape().dim_size(1);
    OP_REQUIRES(context, height > 0 && width > 0,
                errors::InvalidArgument("input size must be positive",
                    input_tensor.shape().DebugString()));

    auto position_flat = position_tensor.flat<float>();
    const float *position = &(position_flat(0));
    auto code_list_flat = code_list_tensor.flat<int32_t>();
    const int32_t *code_list = &(code_list_flat(0));
    auto sortidx_flat = sortidx_tensor.flat<int32_t>();
    const int32_t *sortidx = &(sortidx_flat(0));

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor)); 
    auto out_flat = output_tensor->flat<T>();
    T *out = &(out_flat(0));

    CustomFillAFunctor<Device, T>()(
        context->eigen_device<Device>(),
        position, code_list, count,
        sortidx,
        width, height,
        out);
  }
};


template <typename Device, typename T>
class CustomFillBOp : public OpKernel {
 public:
  explicit CustomFillBOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& position_tensor = context->input(1);
    const Tensor& angle_tensor = context->input(2);
    const Tensor& sortidx_tensor = context->input(3);

    OP_REQUIRES(context, input_tensor.shape().dims() == 3,
                errors::InvalidArgument("input image must be 3-D",
                    input_tensor.shape().DebugString()));
    OP_REQUIRES(context, position_tensor.shape().dims() == 2,
                errors::InvalidArgument("position must be 2-D",
                    position_tensor.shape().DebugString()));
    OP_REQUIRES(context, angle_tensor.shape().dims() == 0,
                errors::InvalidArgument("angle_tensor must be 0-D",
                    angle_tensor.shape().DebugString()));
    OP_REQUIRES(context, sortidx_tensor.shape().dims() == 1,
                errors::InvalidArgument("sortidx must be 1-D",
                    sortidx_tensor.shape().DebugString()));
    OP_REQUIRES(context, input_tensor.shape().dim_size(2) == 4,
                errors::InvalidArgument("input channel must have 4 channel",
                    input_tensor.shape().DebugString()));
    OP_REQUIRES(context, position_tensor.shape().dim_size(1) == 4,
                errors::InvalidArgument("position must have a shape n * 4",
                    position_tensor.shape().DebugString()));

    int count = position_tensor.shape().dim_size(0);
    OP_REQUIRES(context, count == sortidx_tensor.shape().dim_size(0),
                errors::InvalidArgument("Number elements is deferent.",
                  position_tensor.shape().DebugString(),
                  sortidx_tensor.shape().DebugString()));

    int height = input_tensor.shape().dim_size(0);
    int width = input_tensor.shape().dim_size(1);
    OP_REQUIRES(context, height > 0 && width > 0,
                errors::InvalidArgument("input size must be positive",
                    input_tensor.shape().DebugString()));

    auto position_flat = position_tensor.flat<float>();
    const float *position = &(position_flat(0));
    auto angle_flat = angle_tensor.flat<float>();
    const float *angle = &(angle_flat(0));
    auto sortidx_flat = sortidx_tensor.flat<int32_t>();
    const int32_t *sortidx = &(sortidx_flat(0));

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor)); 
    auto out_flat = output_tensor->flat<T>();
    T *out = &(out_flat(0));

    CustomFillBFunctor<Device, T>()(
        context->eigen_device<Device>(),
        position, angle, count,
        sortidx,
        width, height,
        out);
  }
};

template <typename Device, typename T>
class CustomFillCOp : public OpKernel {
 public:
  explicit CustomFillCOp(OpKernelConstruction* context) : OpKernel(context) {}

  void Compute(OpKernelContext* context) override {
    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& position_tensor = context->input(1);

    OP_REQUIRES(context, input_tensor.shape().dims() == 3,
                errors::InvalidArgument("input image must be 3-D",
                    input_tensor.shape().DebugString()));
    OP_REQUIRES(context, position_tensor.shape().dims() == 2,
                errors::InvalidArgument("position must be 2-D",
                    position_tensor.shape().DebugString()));
    OP_REQUIRES(context, input_tensor.shape().dim_size(2) == 1,
                errors::InvalidArgument("input channel must have 1 channel",
                    input_tensor.shape().DebugString()));
    OP_REQUIRES(context, position_tensor.shape().dim_size(1) == 4,
                errors::InvalidArgument("position must have a shape n * 4",
                    position_tensor.shape().DebugString()));

    int count = position_tensor.shape().dim_size(0);

    int height = input_tensor.shape().dim_size(0);
    int width = input_tensor.shape().dim_size(1);
    OP_REQUIRES(context, height > 0 && width > 0,
                errors::InvalidArgument("input size must be positive",
                    input_tensor.shape().DebugString()));

    auto position_flat = position_tensor.flat<float>();
    const float *position = &(position_flat(0));

    // Create an output tensor
    Tensor* output_tensor = NULL;
    OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                     &output_tensor)); 
    auto out_flat = output_tensor->flat<T>();
    T *out = &(out_flat(0));

    CustomFillCFunctor<Device, T>()(
        context->eigen_device<Device>(),
        position, count,
        width, height,
        out);
  }
};

// Register the CPU kernels.
#define REGISTER_CPUA(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CustomFillA").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CustomFillAOp<CPUDevice, T>);
REGISTER_CPUA(int32);
REGISTER_CPUA(int64);

#define REGISTER_CPUB(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CustomFillB").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CustomFillBOp<CPUDevice, T>);
REGISTER_CPUB(float);

#define REGISTER_CPUC(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CustomFillC").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CustomFillCOp<CPUDevice, T>);
REGISTER_CPUC(float);


// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPUA(T)                                          \
  extern template struct CustomFillAFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CustomFillA").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      CustomFillAOp<GPUDevice, T>);
REGISTER_GPUA(int32);
REGISTER_GPUA(int64);

#define REGISTER_GPUB(T)                                          \
  extern template struct CustomFillBFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CustomFillB").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      CustomFillBOp<GPUDevice, T>);
REGISTER_GPUB(float);

#define REGISTER_GPUC(T)                                          \
  extern template struct CustomFillCFunctor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CustomFillC").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      CustomFillCOp<GPUDevice, T>);
REGISTER_GPUC(float);

#endif  // GOOGLE_CUDA
}
}  // namespace tensorflow
