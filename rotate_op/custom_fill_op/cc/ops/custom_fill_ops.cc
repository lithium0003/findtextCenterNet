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

#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"

using namespace tensorflow;

REGISTER_OP("CustomFillA")
    .Attr("T: {int32, int64}")
    .Input("buffer: T")
    .Input("position: float")
    .Input("code_list: int32")
    .Input("sortidx: int32")
    .Output("out: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      ::tensorflow::shape_inference::ShapeHandle position_shape;
      ::tensorflow::shape_inference::ShapeHandle code_list_shape;
      ::tensorflow::shape_inference::ShapeHandle sortidx_shape;
      ::tensorflow::shape_inference::DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &position_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 2, &code_list_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &sortidx_shape));
      TF_RETURN_IF_ERROR(
                c->Merge(c->Dim(position_shape, 0), c->Dim(code_list_shape, 0), &unused));
      TF_RETURN_IF_ERROR(
                c->Merge(c->Dim(position_shape, 0), c->Dim(sortidx_shape, 0), &unused));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(position_shape, 1), 4, &unused));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(code_list_shape, 1), 2, &unused));
      return OkStatus();
    });

REGISTER_OP("CustomFillB")
    .Attr("T: {float}")
    .Input("buffer: T")
    .Input("position: float")
    .Input("angle: float")
    .Input("sortidx: int32")
    .Output("out: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      ::tensorflow::shape_inference::ShapeHandle position_shape;
      ::tensorflow::shape_inference::ShapeHandle sortidx_shape;
      ::tensorflow::shape_inference::ShapeHandle unused1;
      ::tensorflow::shape_inference::DimensionHandle unused2;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &position_shape));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused1));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 1, &sortidx_shape));
      TF_RETURN_IF_ERROR(
                c->Merge(c->Dim(position_shape, 0), c->Dim(sortidx_shape, 0), &unused2));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(position_shape, 1), 4, &unused2));
      return OkStatus();
    });

REGISTER_OP("CustomFillC")
    .Attr("T: {float}")
    .Input("buffer: T")
    .Input("position: float")
    .Output("out: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      c->set_output(0, c->input(0));
      ::tensorflow::shape_inference::ShapeHandle position_shape;
      ::tensorflow::shape_inference::ShapeHandle sortidx_shape;
      ::tensorflow::shape_inference::DimensionHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 2, &position_shape));
      TF_RETURN_IF_ERROR(c->WithValue(c->Dim(position_shape, 1), 4, &unused));
      return OkStatus();
    });
