// kernel_example.h
#ifndef KERNEL_CUSTOM_ROTATE_H_
#define KERNEL_CUSTOM_ROTATE_H_

#include <stdint.h>

namespace tensorflow {

namespace functor {

template <typename Device, typename T>
struct CustomFillAFunctor {
  void operator()(const Device& d, 
    const float *position, const int32_t *code_list, int count,
    const int *sortidx,
    int width, int height,
    T* out);
};

template <typename Device, typename T>
struct CustomFillBFunctor {
  void operator()(const Device& d, 
    const float *position, const float *angle, int count,
    const int *sortidx,
    int width, int height,
    T* out);
};

template <typename Device, typename T>
struct CustomFillCFunctor {
  void operator()(const Device& d, 
    const float *position, int count,
    int width, int height,
    T* out);
};

}  // namespace functor

}  // namespace tensorflow

#endif //KERNEL_CUSTOM_ROTATE_H_
