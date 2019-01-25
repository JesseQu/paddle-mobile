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

#include "operators/kernel/feed_kernel.h"

namespace paddle_mobile {
namespace operators {

inline void save_data_to_file_fp16(std::string file, int64_t number, int64_t channel, int64_t  height, int64_t  width, bool aligned, void* data) {
  int64_t cw = channel * width;
  int64_t aligned_cw = paddle_mobile::fpga::align_to_x(cw, 16);
  int64_t remainder = aligned_cw - cw;

  half* half_data = (half*)data;
  std::ofstream out_file(file.c_str());

  float result = 0.0;
  int64_t index = 0;

  for (int64_t n = 0; n < number; n++){
    for (int64_t h = 0; h < height; h++) {
      for (int64_t i = 0; i < cw; i++) {
        result = paddle_mobile::fpga::fp16_2_fp32(half_data[index]);
        out_file << result << std::endl;
        index++;
      }
      if (aligned) {
        for (int64_t i = 0; i < remainder; i++) {
          index++;
        }
      }
    }
  }
  out_file.close();
}

inline void save_data_to_file_fp32(std::string file, int64_t number, int64_t channel, int64_t  height, int64_t  width, bool aligned, void* data) {
  int64_t cw = channel * width;
  int64_t aligned_cw = paddle_mobile::fpga::align_to_x(cw, 16);
  int64_t remainder = aligned_cw - cw;

  auto* float_data = (float*)data;
  std::ofstream out_file(file.c_str());

  float result = 0.0;
  int64_t index = 0;

  for (int64_t n = 0; n < number; n++){
    for (int64_t h = 0; h < height; h++) {
      for (int64_t i = 0; i < cw; i++) {
        result = float_data[index];
        out_file << result << std::endl;
        index++;
      }
      if (aligned) {
        for (int64_t i = 0; i < remainder; i++) {
          index++;
        }
      }
    }
  }
  out_file.close();
}

template <>
bool FeedKernel<FPGA, float>::Init(FeedParam<FPGA> *param) {
  Tensor *output = param->Out();
  fpga::format_fp16_ofm(output);
  return true;
}

template <>
void FeedKernel<FPGA, float>::Compute(const FeedParam<FPGA> &param) {
  auto input =
      reinterpret_cast<Tensor *>(const_cast<LoDTensor *>(param.InputX()));
  fpga::format_image(input);
  auto input_ptr = input->data<float>();
  Tensor *output = param.Out();
  auto output_ptr = output->data<float>();

  fpga::BypassArgs args = {fpga::DATA_TYPE_FP32};

  args.input_data_type = fpga::DATA_TYPE_FP32;
  args.output_data_type = fpga::DATA_TYPE_FP16;
  args.input_layout_type = fpga::LAYOUT_CHW;
  args.output_layout_type = fpga::LAYOUT_HWC;
  args.image.address = reinterpret_cast<void *>(input_ptr);

  // TODO
  int num = fpga::align_to_x(input->dims()[1] * input->dims()[3], 16) * input->dims()[2];

  args.image.channels = (uint32_t)num;
  args.image.height = (uint32_t)1;
  args.image.width = (uint32_t)1;
  args.image.pad_height = 0;
  args.image.pad_width = 0;
  args.output.address = output_ptr;
  args.output.scale_address = output->scale;
  fpga::PerformBypass(args);

  // save_data_to_file_fp32("feed_input.txt",   input->dims()[0],  input->dims()[1],  input->dims()[2],  input->dims()[3],  input->data_aligned(),  input->data<float>());
  // save_data_to_file_fp16("feed_output.txt", output->dims()[0], output->dims()[1], output->dims()[2], output->dims()[3], output->data_aligned(), output->data<float>());
}
template class FeedKernel<FPGA, float>;

}  // namespace operators
}  // namespace paddle_mobile
