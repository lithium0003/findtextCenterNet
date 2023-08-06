CXX := g++
NVCC := nvcc
PYTHON_BIN_PATH = python

ROTATE_SRCS = custom_rotate_op/cc/kernels/custom_rotate_kernels.cc $(wildcard custom_rotate_op/cc/kernels/*.h) $(wildcard custom_rotate_op/cc/ops/*.cc)
FILL_SRCS = custom_fill_op/cc/kernels/custom_fill_kernels.cc $(wildcard custom_fill_op/cc/kernels/*.h) $(wildcard custom_fill_op/cc/ops/*.cc)

TF_CFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))')
TF_LFLAGS := $(shell $(PYTHON_BIN_PATH) -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))')

CFLAGS = ${TF_CFLAGS} -fPIC -O2 -std=c++11
LDFLAGS = -shared ${TF_LFLAGS}

ROTATE_GPU_ONLY_TARGET_LIB = custom_rotate_op/python/ops/_custom_rotate_op_ops.cu.o
ROTATE_TARGET_LIB = custom_rotate_op/python/ops/_custom_rotate_op_ops.so
FILL_GPU_ONLY_TARGET_LIB = custom_rotate_op/python/ops/_custom_fill_op_ops.cu.o
FILL_TARGET_LIB = custom_rotate_op/python/ops/_custom_fill_op_ops.so

# custom_rotate_op for GPU
rotate_gpu_only: $(ROTATE_GPU_ONLY_TARGET_LIB)

$(ROTATE_GPU_ONLY_TARGET_LIB): custom_rotate_op/cc/kernels/custom_rotate_op.cu.cc
	$(NVCC) -std=c++11 -c -o $@ $^  $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

rotate_op: $(ROTATE_TARGET_LIB)
$(ROTATE_TARGET_LIB): $(ROTATE_SRCS) $(ROTATE_GPU_ONLY_TARGET_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart

rotate_op_test: custom_rotate_op/python/ops/custom_rotate_ops_test.py custom_rotate_op/python/ops/custom_rotate_ops.py $(ROTATE_TARGET_LIB)
	$(PYTHON_BIN_PATH) custom_rotate_op/python/ops/custom_rotate_ops_test.py

# time_two op for GPU
fill_gpu_only: $(FILL_GPU_ONLY_TARGET_LIB)

$(FILL_GPU_ONLY_TARGET_LIB): custom_fill_op/cc/kernels/custom_fill_op.cu.cc
	$(NVCC) -std=c++11 -c -o $@ $^  $(TF_CFLAGS) -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC -DNDEBUG --expt-relaxed-constexpr

fill_op: $(FILL_TARGET_LIB)
$(FILL_TARGET_LIB): $(FILL_SRCS) $(FILL_GPU_ONLY_TARGET_LIB)
	$(CXX) $(CFLAGS) -o $@ $^ ${LDFLAGS}  -D GOOGLE_CUDA=1  -I/usr/local/cuda/targets/x86_64-linux/include -L/usr/local/cuda/targets/x86_64-linux/lib -lcudart

fill_op_test: custom_fill_op/python/ops/custom_fill_ops_test.py custom_fill_op/python/ops/custom_fill_ops.py $(FILL_TARGET_LIB)
	$(PYTHON_BIN_PATH) custom_rotate_op/python/ops/custom_fill_ops_test.py

clean:
	rm -f $(ROTATE_GPU_ONLY_TARGET_LIB) $(ROTATE_TARGET_LIB) $(FILL_GPU_ONLY_TARGET_LIB) $(FILL_TARGET_LIB)
