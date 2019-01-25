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

#ifdef MULTICLASSNMS_OP

#include "operators/kernel/multiclass_nms_kernel.h"
#include "operators/kernel/fpga-kd/multiclass_nms_fpga_func.h"

namespace paddle_mobile {
namespace operators {

void save_to_file(std::string file, int len, float* data) {
  std::ofstream outf(file.c_str());
  for (int i = 0; i < len; i++) {
    // float result = paddle_mobile::fpga::fp16_2_fp32(half_data[i]);
    float result = data[i];
    outf << result << std::endl;
  }
  outf.close();
}

template <>
bool MultiClassNMSKernel<FPGA, float>::Init(MultiClassNMSParam<FPGA> *param) {
  return true;
}

template <>
void MultiClassNMSKernel<FPGA, float>::Compute(
    const MultiClassNMSParam<FPGA> &param) {
  MultiClassNMSCompute<float>(param);
  // auto out = param.Out();
  // // auto out_data = out->data<float>();

  // auto inbox = param.InputBBoxes();
  // auto inscore = param.InputScores();
  // save_to_file("in_box.txt", inbox->numel(), inbox->data<float>());
  // save_to_file("in_score.txt", inscore->numel(), inscore->data<float>());
  // save_to_file("output.txt", out->numel(), out->data<float>());
}

}  // namespace operators
}  // namespace paddle_mobile

#endif
