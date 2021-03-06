/* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License. */

#ifdef RESHAPE_OP

#include "operators/kernel/reshape_kernel.h"

namespace paddle_mobile {
namespace operators {

// TODO move to utility class
static inline void unalign_tensor_data(const Tensor* tensor,half* src, half* dst) {
  int channel = tensor->dims()[1];
  int height = tensor->dims()[2];
  int width = tensor->dims()[3];
  int cw = width * channel;
  int align_cw = fpga::align_to_x(cw, 16); // TODO
  // 把aligned_tensor的有效数据（去掉0）填充到tensor
  for (int h = 0; h < height; h++) {
      memcpy((void *)(dst + h * cw), (void *)(src + h * align_cw), 
           cw * sizeof(half));
  }
}

template <>
bool ReshapeKernel<FPGA, float>::Init(ReshapeParam<FPGA> *param) {
  return true;
}

template <>
void ReshapeKernel<FPGA, float>::Compute(const ReshapeParam<FPGA> &param) {
//  if (true) {
//    return;
//  }
  const auto *input_x = param.InputX();
  const auto &input_x_dims = input_x->dims();
  auto *out = param.Out();
  out->set_data_aligned(input_x->data_aligned());
  framework::DDim out_dims = out->dims();
  const auto *input_shape = param.InputShape();

  if (input_shape) {
    auto *shape_data = input_shape->data<int>();
    framework::Tensor cpu_shape_tensor;
    auto shape =
        std::vector<int>(shape_data, shape_data + input_shape->numel());
    out_dims = ValidateShape(shape, input_x->dims());
  }

  bool inplace = param.Inplace();

  out->Resize(out_dims);
  if (!inplace) {
    out->mutable_data<float>();
    framework::TensorCopy(*input_x, out);  // TODO(chonwhite) is it right?
    out->Resize(out_dims);
  } else {
    out->ShareDataWith(*input_x);
    out->Resize(out_dims);
  }

  auto input = param.InputX();
  // DLOG << "reshape: aligned:" << input->data_aligned() << "  " << out->dims();
  if (input->dims().size() == out->dims().size()) {
    return;
  }
  if (!input->data_aligned()) {
    return;
  }
  out->set_data_aligned(false);
  auto output = param.Out();
  half* aligned_data = (half*)input->data<float>();
  // half* unaligned_data = (half*)fpga::fpga_malloc(output->numel() * sizeof(half));
  half* unaligned_data = (half*)output->data<float>();

  unalign_tensor_data(input, aligned_data, unaligned_data);

//  out->set_data_aligned(false);
//  auto output = param.Out();
//  half* aligned_data = (half*)input->data<float>();
//  half* unaligned_data = (half*)fpga::fpga_malloc(output->numel() * sizeof(half));
//  unalign_tensor_data(input, aligned_data, unaligned_data);
//  output->reset_data_ptr(unaligned_data);// TODO align from input;
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
