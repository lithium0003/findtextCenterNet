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

#include "custom_rotate.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/shape_inference.h"

namespace tensorflow {

typedef Eigen::ThreadPoolDevice CPUDevice;
typedef Eigen::GpuDevice GPUDevice;

namespace functor {
// CPU specialization of actual computation.
template <typename T>
struct CustomRotate1Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d,
    int im_width, int im_height, int im_ch, 
    int width, int height,
    const float *rot_cx, const float *rot_cy,
    const float *sx, const float *sy, const float *angle,
    const T* in, float* out) {
      
      for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
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
  }
};

template <typename T>
struct CustomRotate2Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d,
    int im_width, int im_height, int im_ch, 
    int width, int height,
    const float *rot_cx, const float *rot_cy,
    const float *sx, const float *sy, const float *angle,
    const T* in, T* out) {
      
      for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
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
  }
};

template <typename T>
struct CustomRotate3Functor<CPUDevice, T> {
  void operator()(const CPUDevice& d,
    int im_width, int im_height, int im_ch, 
    int width, int height,
    const float *rot_cx, const float *rot_cy,
    const float *sx, const float *sy, const float *angle,
    const T* in, float* out) {
      
      for(int y = 0; y < height; y++) {
        for(int x = 0; x < width; x++) {
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
  }
};


// OpKernel definition.
// template parameter <T> is the datatype of the tensors.
template <typename Device, typename T>
class CustomRotate1Op : public OpKernel {
 public:
  explicit CustomRotate1Op(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("width", &width_));
    OP_REQUIRES_OK(context, context->GetAttr("height", &height_));
  }

  void Compute(OpKernelContext* context) override {
    int im_width_;
    int im_height_;
    int im_ch_;

    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& rot_cx_tensor = context->input(1);
    const Tensor& rot_cy_tensor = context->input(2);
    const Tensor& sx_tensor = context->input(3);
    const Tensor& sy_tensor = context->input(4);
    const Tensor& angle_tensor = context->input(5);

    const float *rot_cx_ = rot_cx_tensor.flat<float>().data();
    const float *rot_cy_ = rot_cy_tensor.flat<float>().data();
    const float *sx_ = sx_tensor.flat<float>().data();
    const float *sy_ = sy_tensor.flat<float>().data();
    const float *angle_ = angle_tensor.flat<float>().data();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    im_height_ = input_tensor.shape().dim_size(0);
    im_width_ = input_tensor.shape().dim_size(1);
    if(input_tensor.shape().dims() > 2) {
      im_ch_ = input_tensor.shape().dim_size(2);
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({ height_, width_, im_ch_ }),
                                                      &output_tensor));
    }
    else {
      im_ch_ = 1;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({ height_, width_}),
                                                      &output_tensor));
    }
    auto in_flat = input_tensor.flat<T>();
    const T *in = &(in_flat(0));
    auto out_flat = output_tensor->flat<float>();
    float *out = &(out_flat(0));

    CustomRotate1Functor<Device, T>()(
        context->eigen_device<Device>(),
        im_width_, im_height_, im_ch_,
        width_, height_,
        rot_cx_, rot_cy_,
        sx_, sy_, angle_,
        in,
        out);
  }
 private:
  int width_;
  int height_;
};

template <typename Device, typename T>
class CustomRotate2Op : public OpKernel {
 public:
  explicit CustomRotate2Op(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("width", &width_));
    OP_REQUIRES_OK(context, context->GetAttr("height", &height_));
  }

  void Compute(OpKernelContext* context) override {
    int im_width_;
    int im_height_;
    int im_ch_;

    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& rot_cx_tensor = context->input(1);
    const Tensor& rot_cy_tensor = context->input(2);
    const Tensor& sx_tensor = context->input(3);
    const Tensor& sy_tensor = context->input(4);
    const Tensor& angle_tensor = context->input(5);

    const float *rot_cx_ = rot_cx_tensor.flat<float>().data();
    const float *rot_cy_ = rot_cy_tensor.flat<float>().data();
    const float *sx_ = sx_tensor.flat<float>().data();
    const float *sy_ = sy_tensor.flat<float>().data();
    const float *angle_ = angle_tensor.flat<float>().data();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    im_height_ = input_tensor.shape().dim_size(0);
    im_width_ = input_tensor.shape().dim_size(1);
    if(input_tensor.shape().dims() > 2) {
      im_ch_ = input_tensor.shape().dim_size(2);
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({ height_, width_, im_ch_ }),
                                                      &output_tensor));
    }
    else {
      im_ch_ = 1;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({ height_, width_}),
                                                      &output_tensor));
    }
    auto in_flat = input_tensor.flat<T>();
    const T *in = &(in_flat(0));
    auto out_flat = output_tensor->flat<T>();
    T *out = &(out_flat(0));

    CustomRotate2Functor<Device, T>()(
        context->eigen_device<Device>(),
        im_width_, im_height_, im_ch_,
        width_, height_,
        rot_cx_, rot_cy_,
        sx_, sy_, angle_,
        in,
        out);
  }
 private:
  int width_;
  int height_;
};

template <typename Device, typename T>
class CustomRotate3Op : public OpKernel {
 public:
  explicit CustomRotate3Op(OpKernelConstruction* context) : OpKernel(context) {
    OP_REQUIRES_OK(context, context->GetAttr("width", &width_));
    OP_REQUIRES_OK(context, context->GetAttr("height", &height_));
  }

  void Compute(OpKernelContext* context) override {
    int im_width_;
    int im_height_;
    int im_ch_;

    // Grab the input tensor
    const Tensor& input_tensor = context->input(0);
    const Tensor& rot_cx_tensor = context->input(1);
    const Tensor& rot_cy_tensor = context->input(2);
    const Tensor& sx_tensor = context->input(3);
    const Tensor& sy_tensor = context->input(4);
    const Tensor& angle_tensor = context->input(5);

    const float *rot_cx_ = rot_cx_tensor.flat<float>().data();
    const float *rot_cy_ = rot_cy_tensor.flat<float>().data();
    const float *sx_ = sx_tensor.flat<float>().data();
    const float *sy_ = sy_tensor.flat<float>().data();
    const float *angle_ = angle_tensor.flat<float>().data();

    // Create an output tensor
    Tensor* output_tensor = NULL;
    im_height_ = input_tensor.shape().dim_size(0);
    im_width_ = input_tensor.shape().dim_size(1);
    if(input_tensor.shape().dims() > 2) {
      im_ch_ = input_tensor.shape().dim_size(2);
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({ height_, width_, im_ch_ }),
                                                      &output_tensor));
    }
    else {
      im_ch_ = 1;
      OP_REQUIRES_OK(context, context->allocate_output(0, TensorShape({ height_, width_}),
                                                      &output_tensor));
    }
    auto in_flat = input_tensor.flat<T>();
    const T *in = &(in_flat(0));
    auto out_flat = output_tensor->flat<float>();
    float *out = &(out_flat(0));

    CustomRotate3Functor<Device, T>()(
        context->eigen_device<Device>(),
        im_width_, im_height_, im_ch_,
        width_, height_,
        rot_cx_, rot_cy_,
        sx_, sy_, angle_,
        in,
        out);
  }
 private:
  int width_;
  int height_;
};

// Register the CPU kernels.
#define REGISTER_CPU1(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CustomRotate1").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CustomRotate1Op<CPUDevice, T>);
REGISTER_CPU1(float);
REGISTER_CPU1(double);
REGISTER_CPU1(int32);
REGISTER_CPU1(int64);
REGISTER_CPU1(uint8);
REGISTER_CPU1(int16);
REGISTER_CPU1(int8);
REGISTER_CPU1(uint16);
REGISTER_CPU1(uint32);
REGISTER_CPU1(uint64);

#define REGISTER_CPU2(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CustomRotate2").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CustomRotate2Op<CPUDevice, T>);
REGISTER_CPU2(float);
REGISTER_CPU2(double);
REGISTER_CPU2(int32);
REGISTER_CPU2(int64);
REGISTER_CPU2(uint8);
REGISTER_CPU2(int16);
REGISTER_CPU2(int8);
REGISTER_CPU2(uint16);
REGISTER_CPU2(uint32);
REGISTER_CPU2(uint64);

#define REGISTER_CPU3(T)                                          \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CustomRotate3").Device(DEVICE_CPU).TypeConstraint<T>("T"), \
      CustomRotate3Op<CPUDevice, T>);
REGISTER_CPU3(float);
REGISTER_CPU3(double);
REGISTER_CPU3(int32);
REGISTER_CPU3(int64);
REGISTER_CPU3(uint8);
REGISTER_CPU3(int16);
REGISTER_CPU3(int8);
REGISTER_CPU3(uint16);
REGISTER_CPU3(uint32);
REGISTER_CPU3(uint64);

// Register the GPU kernels.
#ifdef GOOGLE_CUDA
#define REGISTER_GPU1(T)                                          \
  extern template struct CustomRotate1Functor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CustomRotate1").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      CustomRotate1Op<GPUDevice, T>);
REGISTER_GPU1(float);
REGISTER_GPU1(double);
REGISTER_GPU1(int32);
REGISTER_GPU1(int64);
REGISTER_GPU1(uint8);
REGISTER_GPU1(int16);
REGISTER_GPU1(int8);
REGISTER_GPU1(uint16);
REGISTER_GPU1(uint32);
REGISTER_GPU1(uint64);

#define REGISTER_GPU2(T)                                          \
  extern template struct CustomRotate2Functor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CustomRotate2").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      CustomRotate2Op<GPUDevice, T>);
REGISTER_GPU2(float);
REGISTER_GPU2(double);
REGISTER_GPU2(int32);
REGISTER_GPU2(int64);
REGISTER_GPU2(uint8);
REGISTER_GPU2(int16);
REGISTER_GPU2(int8);
REGISTER_GPU2(uint16);
REGISTER_GPU2(uint32);
REGISTER_GPU2(uint64);

#define REGISTER_GPU3(T)                                          \
  extern template struct CustomRotate1Functor<GPUDevice, T>;           \
  REGISTER_KERNEL_BUILDER(                                       \
      Name("CustomRotate3").Device(DEVICE_GPU).TypeConstraint<T>("T"), \
      CustomRotate3Op<GPUDevice, T>);
REGISTER_GPU3(float);
REGISTER_GPU3(double);
REGISTER_GPU3(int32);
REGISTER_GPU3(int64);
REGISTER_GPU3(uint8);
REGISTER_GPU3(int16);
REGISTER_GPU3(int8);
REGISTER_GPU3(uint16);
REGISTER_GPU3(uint32);
REGISTER_GPU3(uint64);

#endif  // GOOGLE_CUDA
}
}  // namespace tensorflow
