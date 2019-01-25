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

#ifdef SOFTMAX_OP

#include "operators/kernel/softmax_kernel.h"
#include "operators/kernel/fpga-kd/softmax_fpga_func.h"

namespace paddle_mobile {
namespace operators {

inline void save_to_file(std::string file, int len, float* data) {
  std::ofstream outf(file.c_str());
  for (int i = 0; i < len; i++) {
    // float result = paddle_mobile::fpga::fp16_2_fp32(half_data[i]);
    float result = data[i];
    outf << result << std::endl;
  }
  outf.close();
}

template <>
bool SoftmaxKernel<FPGA, float>::Init(SoftmaxParam<FPGA> *param) {
  auto input = const_cast<Tensor *>(param->InputX());
  auto input_ptr = input->data<float>();
  auto out = param->Out();

  // DLOG << "out dim::" << out->dims();

  // fpga::format_fp32_ofm(out);

  auto float_input = new Tensor;
  float_input->mutable_data<float>(input->dims());
  // fpga::format_fp32_ofm(float_input);

  fpga::BypassArgs args = {fpga::DATA_TYPE_FP16};
  args.input_layout_type = fpga::LAYOUT_HWC;
  args.output_layout_type = fpga::LAYOUT_CHW;
  args.input_data_type = fpga::DATA_TYPE_FP16;
  args.output_data_type = fpga::DATA_TYPE_FP32;
  args.image.address = input_ptr;
  args.image.height = (uint32_t)1;
  args.image.width = (uint32_t)1;
  args.image.channels = (uint32_t)input->numel();
  args.output.address = float_input->data<float>();
  args.output.scale_address = float_input->scale;
  param->SetFloatInput(float_input);
  param->SetFpgaArgs(args);
  return true;
}

template <>
void SoftmaxKernel<FPGA, float>::Compute(const SoftmaxParam<FPGA> &param) {
  Tensor *in_x = param.FloatInput();
  Tensor *out = param.Out();
  out->set_data_aligned(false);

  DLOG << "before bypass";
  fpga::PerformBypass(param.FpgaArgs());
  fpga::fpga_invalidate((void *)in_x->data<float>(),  // NOLINT
                        in_x->numel() * sizeof(float));

  DLOG << "after bypass";
  // TODO: In general case, 0 should be squeezed before softmax input
  math::SoftmaxFuntor<float>()(in_x, out);
  DLOG << "after SoftmaxFuntor";
  fpga::fpga_flush(out->data<float>(), out->memory_size());

  // save_to_file("soft_input.txt", in_x->numel(), in_x->data<float>());
  // save_to_file("soft_out.txt", out->numel(), out->data<float>());
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
