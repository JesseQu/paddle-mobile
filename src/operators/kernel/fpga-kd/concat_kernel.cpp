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

#ifdef CONCAT_OP

#include "operators/kernel/concat_kernel.h"

namespace paddle_mobile {
namespace operators {

void concat3D(const ConcatParam<FPGA> &param) {

  // DLOG << "concat3D";
  auto input = param.Inputs();
  auto output = param.Out();
  int axis = param.Axis();
  // DLOG << "concat::: data_aligned:" << input[0]->data_aligned();
  output->set_data_aligned(input[0]->data_aligned());

  size_t num = input.size();
  int rows = 1;
  auto dim_0 = input[0]->dims();
  for (int i = 0; i < axis; ++i) {
    rows *= dim_0[i];
  }
  int out_rows = rows, out_cols = 0;

  std::vector<int64_t> input_cols(input.size());
  for (int i = 0; i < num; ++i) {
    int t_cols = input[i]->numel() / rows;
    out_cols += t_cols;
    input_cols[i] = t_cols;
  }

  // computation
  for (int k = 0; k < out_rows; ++k) {
    half* dst_ptr = ((half*)output->data<float>()) + k * out_cols;
    int col_idx = 0;
    for (int j = 0; j < num; ++j) {
      int col_len = input_cols[j];
      const half* src_prt = ((half*)input[j]->data<float>()) + k * col_len;
      memory::Copy(dst_ptr + col_idx, src_prt, sizeof(half) * col_len);
      col_idx += col_len;
    }
  }
}

void concat2D(const ConcatParam<FPGA> &param) {

}

template <>
bool ConcatKernel<FPGA, float>::Init(ConcatParam<FPGA> *param) {
  auto inputs = param->Inputs();
  auto out = param->Out();
  auto image_num = inputs.size();
  auto images_in =
      (half **)fpga::fpga_malloc(image_num * sizeof(int *));  // NOLINT
  auto scales_in =
      (float **)fpga::fpga_malloc(image_num * sizeof(float *));  // NOLINT
  auto channel_num =
      (uint32_t *)fpga::fpga_malloc(image_num * sizeof(uint32_t));  // NOLINT

  // DLOG << "concat_size:" << inputs[0]->dims().size();
  // DLOG << "concat_size:" << inputs[0]->dims();
  if (inputs[0]->dims().size() <= 3) {
    return true;
  }

  auto height = inputs[0]->dims()[2];
  auto width = inputs[0]->dims()[3];
  for (int i = 0; i < image_num; i++) {
    auto input = inputs[i];
    PADDLE_MOBILE_ENFORCE(
        input->dims()[2] == height && input->dims()[3] == width,
        "Image height & width should be unified");
    images_in[i] = (half *)input->data<float>();      // NOLINT
    channel_num[i] = (uint32_t)inputs[i]->dims()[1];  // NOLINT
    scales_in[i] = input->scale;
  }
  fpga::format_concat_output(out, height, width, image_num, channel_num);

  fpga::ConcatArgs concatArgs = {0};
  concatArgs.image_num = image_num;
  concatArgs.images_in = images_in;
  concatArgs.scales_in = scales_in;
  concatArgs.image_out = (half *)out->data<float>();  // NOLINT
  concatArgs.scale_out = out->scale;
  concatArgs.channel_num = channel_num;
  concatArgs.height = height;
  concatArgs.width = width;
  param->SetFpgaArgs(concatArgs);
  return true;
}

template <>
void ConcatKernel<FPGA, float>::Compute(const ConcatParam<FPGA> &param) {
  auto inputs = param.Inputs();
  int dim_size = inputs[0]->dims().size();
  if (dim_size <= 3) {
    if (dim_size == 3) {
      concat3D(param);
    }
    if (dim_size == 2) {
      concat3D(param);
    }
    return; // TODO
  }
  ComputeFPGAConcat(param.FpgaArgs());
}
template class ConcatKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile

#endif
