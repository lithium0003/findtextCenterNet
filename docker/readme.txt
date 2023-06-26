sudo docker build -t tensorflow-custom-op:ubuntu22 -f docker/ubuntu22.dockerfile docker

docker run --gpus all -it --rm -v ${PWD}:/working_dir -w /working_dir tensorflow-custom-op:ubuntu22

(docker)#  cd rotate_op
(docker)#  ./configure.sh

(docker)#  bazel build build_pip_pkg
(docker)#  bazel-bin/build_pip_pkg artifacts

(docker)#  bazel test custom_rotate_op:custom_rotate_ops_py_test
(docker)#  bazel test custom_fill_op:custom_fill_ops_py_test

(docker)#  bazel clean --expunge


docker image prune -a
docker image list


from custom_rotate_op import biliner_rotate_ops
biliner_rotate_ops([[1, 2], [3, 4]], width=2, height=2, rot_cx=0.5, rot_cy=0.5, sx=1., sy=1., angle=0.)