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

REGISTER_OP("CustomRotate1")
    .Attr("T: realnumbertype")
    .Attr("width: int >= 1")
    .Attr("height: int >= 1")
    .Input("image: T")
    .Input("rot_cx: float")
    .Input("rot_cy: float")
    .Input("sx: float")
    .Input("sy: float")
    .Input("angle: float")
    .Output("out: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      if (c->num_inputs() != 6) {
        return errors::InvalidArgument("Expected 6 input but got: ",
                                      c->num_inputs());
      }
      int width, height;
      TF_RETURN_IF_ERROR(c->GetAttr("width", &width));
      TF_RETURN_IF_ERROR(c->GetAttr("height", &height));
      ::tensorflow::shape_inference::ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 3, &input_shape));
      ::tensorflow::shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      if(c->Rank(input_shape) == 3) {
        c->set_output(0, c->MakeShape({height, width, c->Value(c->Dim(input_shape, 2))}));
      }
      else {
        c->set_output(0, c->MakeShape({height, width}));
      }
      return OkStatus();
    });

REGISTER_OP("CustomRotate2")
    .Attr("T: realnumbertype")
    .Attr("width: int >= 1")
    .Attr("height: int >= 1")
    .Input("image: T")
    .Input("rot_cx: float")
    .Input("rot_cy: float")
    .Input("sx: float")
    .Input("sy: float")
    .Input("angle: float")
    .Output("out: T")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      if (c->num_inputs() != 6) {
        return errors::InvalidArgument("Expected 6 input but got: ",
                                      c->num_inputs());
      }
      int width, height;
      TF_RETURN_IF_ERROR(c->GetAttr("width", &width));
      TF_RETURN_IF_ERROR(c->GetAttr("height", &height));
      ::tensorflow::shape_inference::ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 3, &input_shape));
      ::tensorflow::shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      if(c->Rank(input_shape) == 3) {
        c->set_output(0, c->MakeShape({height, width, c->Value(c->Dim(input_shape, 2))}));
      }
      else {
        c->set_output(0, c->MakeShape({height, width}));
      }
      return OkStatus();
    });

REGISTER_OP("CustomRotate3")
    .Attr("T: realnumbertype")
    .Attr("width: int >= 1")
    .Attr("height: int >= 1")
    .Input("image: T")
    .Input("rot_cx: float")
    .Input("rot_cy: float")
    .Input("sx: float")
    .Input("sy: float")
    .Input("angle: float")
    .Output("out: float")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext* c) {
      if (c->num_inputs() != 6) {
        return errors::InvalidArgument("Expected 6 input but got: ",
                                      c->num_inputs());
      }
      int width, height;
      TF_RETURN_IF_ERROR(c->GetAttr("width", &width));
      TF_RETURN_IF_ERROR(c->GetAttr("height", &height));
      ::tensorflow::shape_inference::ShapeHandle input_shape;
      TF_RETURN_IF_ERROR(c->WithRankAtMost(c->input(0), 3, &input_shape));
      ::tensorflow::shape_inference::ShapeHandle unused;
      TF_RETURN_IF_ERROR(c->WithRank(c->input(1), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(2), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(3), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(4), 0, &unused));
      TF_RETURN_IF_ERROR(c->WithRank(c->input(5), 0, &unused));
      if(c->Rank(input_shape) == 3) {
        c->set_output(0, c->MakeShape({height, width, c->Value(c->Dim(input_shape, 2))}));
      }
      else {
        c->set_output(0, c->MakeShape({height, width}));
      }
      return OkStatus();
    });
