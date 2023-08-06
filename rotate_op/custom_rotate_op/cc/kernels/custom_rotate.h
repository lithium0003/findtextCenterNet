// kernel_example.h
#ifndef KERNEL_CUSTOM_ROTATE_H_
#define KERNEL_CUSTOM_ROTATE_H_

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct CustomRotate1Functor {
  void operator()(const Device& d, 
    int im_width, int im_height, int im_ch, 
    int width, int height,
    const float *rot_cx, const float *rot_cy,
    const float *sx, const float *sy, const float *angle,
    const T* in, float* out);
};

template <typename Device, typename T>
struct CustomRotate2Functor {
  void operator()(const Device& d, 
    int im_width, int im_height, int im_ch, 
    int width, int height,
    const float *rot_cx, const float *rot_cy,
    const float *sx, const float *sy, const float *angle,
    const T* in, T* out);
};

template <typename Device, typename T>
struct CustomRotate3Functor {
  void operator()(const Device& d, 
    int im_width, int im_height, int im_ch, 
    int width, int height,
    const float *rot_cx, const float *rot_cy,
    const float *sx, const float *sy, const float *angle,
    const T* in, float* out);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_CUSTOM_ROTATE_H_
