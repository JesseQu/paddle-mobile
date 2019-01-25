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

#ifdef SPLIT_OP

#include "operators/kernel/split_kernel.h"

namespace paddle_mobile {
namespace operators {


template <typename T>
inline void StridedNumelCopyWithAxis(int64_t axis, T* dst,
                                     const framework::DDim& dst_stride_numel,
                                     const T* src,
                                     const framework::DDim& src_stride_numel,
                                     int64_t size) {
  int64_t before = dst_stride_numel[0] / dst_stride_numel[axis];
  int64_t src_after = src_stride_numel[axis];
  int64_t dst_after = dst_stride_numel[axis];

  // DLOG << "before:" << before;
  // DLOG << "src_after:" << src_after;
  // DLOG << "dst_after:" << dst_after;

  PADDLE_MOBILE_ENFORCE(src_stride_numel.size() == dst_stride_numel.size(),
                        "src and dst tensor should have the same dims size.");

  for (int64_t i = 0; i < axis; ++i) {
    if (i < axis) {
      PADDLE_MOBILE_ENFORCE(src_stride_numel[i] / src_stride_numel[axis] ==
                                dst_stride_numel[i] / dst_stride_numel[axis],
                            "src and dst should have the same elements "
                            "except the specified axis.");
    } else if (i == axis) {
      continue;
    } else {
      PADDLE_MOBILE_ENFORCE(src_stride_numel[i] == dst_stride_numel[i],
                            "src and dst should have the same elements "
                            "except the specified axis.");
    }
  }

  for (int64_t i = 0; i < before; ++i) {
    memory::Copy(dst + i * dst_after, src + i * src_after, sizeof(T) * size);
  }
}

template <>
bool SplitKernel<FPGA, float>::Init(SplitParam<FPGA> *param) {
  auto *in = const_cast<Tensor *>(param->InputX());
  auto outs = param->Outs();
  auto sections = param->Sections();
  int axis = param->Axis();

  // DLOG << "outs.size():" << outs.size();
  // DLOG << "sections.size():" << sections.size();

  for (auto out: outs) {
    out->set_data_aligned(in->data_aligned());
  }

  if (in->dims().size() <= 3) {
    return true;
  }

  PADDLE_MOBILE_ENFORCE(axis == 1, "Only support split in channel dimension");
  if (sections.size() != 0) {
    PADDLE_MOBILE_ENFORCE(outs.size() == sections.size(),
                        "Output number should be equal to section number");
  }

  auto image_num = (uint32_t)outs.size();
  auto images_out =
      reinterpret_cast<void **>(fpga::fpga_malloc(image_num * sizeof(void *)));
  auto scales_out = reinterpret_cast<float **>(
      fpga::fpga_malloc(image_num * sizeof(float *)));
  auto out_channels = reinterpret_cast<uint32_t *>(
      fpga::fpga_malloc(image_num * sizeof(uint32_t)));
  for (int i = 0; i < image_num; i++) {
    fpga::format_fp16_ofm(outs[i]);
    images_out[i] = outs[i]->mutable_data<float>();
    scales_out[i] = outs[i]->scale;
    if (sections.size() == 0) {
      out_channels[i] = outs[0]->dims()[1];
    } else{
      out_channels[i] = (uint32_t)sections[i];
    }
  }
  int width = 1;
  int height = 1;
  if (in->dims().size() > 3) {
    width = in->dims()[3];
  }

  if (in->dims().size() > 2) {
    height = in->dims()[2];
  }

  fpga::SplitArgs arg = {0};
  arg.image_num = image_num;
  arg.image_in = (half *)in->data<float>();
  arg.scale_in = in->scale;
  arg.images_out = images_out;
  arg.scales_out = scales_out;
  arg.out_channel_nums = out_channels;
  arg.height = height;
  arg.width = width;

  param->SetFpgaArgs(arg);

  // auto *in = const_cast<Tensor *>(param.InputX());
  // auto outs = param.Outs();

  return true;
}
template <>
void SplitKernel<FPGA, float>::Compute(const SplitParam<FPGA> &param) {
  auto *in = const_cast<Tensor *>(param.InputX());
  auto outs = param.Outs();
  for (auto out: outs) {
    out->set_data_aligned(in->data_aligned());
  }

  if (in->dims().size() <= 3) {
    auto in_stride = framework::stride_numel(in->dims());
    int64_t axis = param.Axis();
    size_t input_offset = 0;
    half* in_data = (half*)in->data<float>();

    for (auto& out : outs) {

      half* out_data = (half*)out->mutable_data<float>();
      // out->mutable_data<float>();
      auto out_stride = framework::stride_numel(out->dims());
      // DLOG << "out_stride::" << out_stride;

      StridedNumelCopyWithAxis<half>(axis, out_data, out_stride,
                                      in_data + input_offset, in_stride,
                                      out_stride[axis]);
      input_offset += out_stride[axis];
      
    }

    // exit(-1);
    return; // TODO
  }
  fpga::ComputeFPGASplit(param.FpgaArgs());
}

}  // namespace operators
}  // namespace paddle_mobile
#endif
