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

#include "common/types.h"

#include "fpga-kd/pe.h"
#include "fpga-kd/filter.h"
#include "fpga-kd/image.h"

#ifdef COST_TIME_PRINT
#include <sys/time.h>
#include <time.h>
#include <iomanip>
#include <iostream>
#endif

namespace paddle_mobile {
namespace fpga {

using namespace std;     // NOLINT

int ComputeFpgaConv(const struct SplitConvArgs &args) {
//  ComputeBasicConv(args.conv_arg[0]);
#ifdef FPGA_PRINT_MODE
  DLOG << "=============ComputeFPGAConv===========";
  DLOG << "   filter_num:" << args.filter_num
       << "   group_num:" << args.group_num
       << "   split_num:" << args.split_num;
#endif
  int ret = 0;
  int split_num = args.split_num;
  for (int i = 0; i < split_num; i++) {
    ret |= ComputeBasicConv(args.conv_arg[i]);
  }

  if (split_num > 1) {
    ComputeFPGAConcat(args.concat_arg);
  }

  return ret;
}

int ComputeBasicConv(const struct ConvArgs &args) {
#ifdef FPGA_PRINT_MODE
  DLOG << "======Compute Basic Conv======";
  DLOG << "   relu_enabled:" << args.relu_enabled
       << "   sb_address:" << args.sb_address
       << "   filter_address:" << args.filter_address
       << "   filter_num:" << args.filter_num
       << "   group_num:" << args.group_num;
  DLOG << "   image_address:" << args.image.address
       << "   image_scale_address:" << args.image.scale_address
       << "   image_channels:" << args.image.channels
       << "   image_height:" << args.image.height
       << "   image_width:" << args.image.width
       << "   pad_height:" << args.image.pad_height
       << "   pad_width:" << args.image.pad_width;
  DLOG << "   kernel_height:" << args.kernel.height
       << "   kernel_width:" << args.kernel.width
       << "   stride_h:" << args.kernel.stride_h
       << "   stride_w:" << args.kernel.stride_w;
  DLOG << "   out_address:" << args.output.address
       << "   out_scale_address:" << args.output.scale_address;
#endif

  return compute_fpga_conv_basic(args);

}  // ComputeBasicConv

int ComputeFpgaPool(const struct PoolingArgs &args) {
#ifdef FPGA_PRINT_MODE
  DLOG << "=============ComputeFpgaPool===========";
  DLOG << "   mode:" << args.mode
       << "   kernel_reciprocal:" << fp16_2_fp32(args.kernel_reciprocal);
  DLOG << "   image_address:" << args.image.address
       << "   image_scale_address:" << args.image.scale_address
       << "   image_channels:" << args.image.channels
       << "   image_height:" << args.image.height
       << "   image_width:" << args.image.width
       << "   pad_height:" << args.image.pad_height
       << "   pad_width:" << args.image.pad_width;
  DLOG << "   kernel_height:" << args.kernel.height
       << "   kernel_width:" << args.kernel.width
       << "   stride_h:" << args.kernel.stride_h
       << "   stride_w:" << args.kernel.stride_w;
  DLOG << "   out_address:" << args.output.address
       << "   out_scale_address:" << args.output.scale_address;
#endif
  return compute_fpga_pool(args);
}  // ComputeFpgaPool

int ComputeFpgaEWAdd(const struct EWAddArgs &args) {
#ifdef FPGA_PRINT_MODE
  DLOG << "=============ComputeFpgaEWAdd===========";
  DLOG << "   relu_enabled:" << args.relu_enabled
       << "   const0:" << fp16_2_fp32(int16_t(args.const0))
       << "   const1:" << fp16_2_fp32(int16_t(args.const1));
  DLOG << "   image0_address:" << args.image0.address
       << "   image0_scale_address:" << args.image0.scale_address
       << "   image0_channels:" << args.image0.channels
       << "   image0_height:" << args.image0.height
       << "   image0_width:" << args.image0.width
       << "   pad0_height:" << args.image0.pad_height
       << "   pad0_width:" << args.image0.pad_width;
  DLOG << "   image1_address:" << args.image1.address
       << "   image1_scale_address:" << args.image1.scale_address
       << "   image1_channels:" << args.image1.channels
       << "   image1_height:" << args.image1.height
       << "   image1_width:" << args.image1.width
       << "   pad1_height:" << args.image1.pad_height
       << "   pad_width:" << args.image1.pad_width;
  DLOG << "   out_address:" << args.output.address
       << "   out_scale_address:" << args.output.scale_address;
#endif
  return compute_fpga_ewadd(args);
}  // ComputeFpgaEWAdd

int PerformBypass(const struct BypassArgs &args) {
#ifdef FPGA_PRINT_MODE
  DLOG << "=============ComputeFpgaBypass===========";
  DLOG << "   input_type:" << args.input_data_type
       << "   output_type:" << args.output_data_type
       << "   input_layout_type:" << args.input_layout_type
       << "   output_layout_type:" << args.output_layout_type;
  DLOG << "   image_address:" << args.image.address
       << "   image_scale_address:" << args.image.scale_address
       << "   image_channels:" << args.image.channels
       << "   image_height:" << args.image.height
       << "   image_width:" << args.image.width
       << "   pad_height:" << args.image.pad_height
       << "   pad_width:" << args.image.pad_width;
  DLOG << "   out_address:" << args.output.address
       << "   out_scale_address:" << args.output.scale_address;
#endif
  return perform_bypass(args);
}  // PerformBypass

int ComputeFPGAConcat(const struct ConcatArgs &args) {
#ifdef FPGA_PRINT_MODE
  DLOG << "=============ComputeFpgaConcat===========";
  DLOG << "   Image_num: " << args.image_num
       << "   out_address:" << args.image_out
       << "   out_scale_address:" << args.scale_out
       << "   out_channel:" << args.out_channel;
  DLOG << "   image_height:" << args.height << "   image_width:" << args.width;
  for (int i = 0; i < args.image_num; i++) {
    DLOG << "   " << i << "th:        ";
    DLOG << "   channel_num:"
         << args.channel_num[i]
         //<< "   aligned_channel_num:" << args.aligned_channel_num[i]
         << "   image_address:" << args.images_in[i]
         << "   image_scale_address:" << args.scales_in[i];
  }
#endif

  image::concat_images(args.images_in, args.scales_in, args.image_out,
                       args.scale_out, args.image_num, args.channel_num,
                       args.height, args.width);
  return 0;
}  // ComputeFPGAConcat

void deconv_post_process(const struct DeconvArgs &args) {
  int sub_conv_n = args.sub_conv_num;
  int sub_height = args.sub_output_height;
  int sub_width = args.sub_output_width;
  int omit_size = args.omit_size;
  int channel = args.filter_num;
  int num = 1;
  int origin_h = sub_height * sub_conv_n;
  int origin_w = sub_width * sub_conv_n;
  int align_origin_w = align_to_x(origin_w * channel, 16);
  int deconv_h = origin_h - 2 * omit_size;
  int deconv_w = origin_w - 2 * omit_size;
  int deconv_row_len = deconv_w * channel;
  int align_deconv_row_len = align_to_x(deconv_row_len, 16);

  for (int idx = 0; idx < sub_conv_n; ++idx) {
    paddle_mobile::fpga::fpga_invalidate(
        args.split_conv_args[idx].output.address,
        align_origin_w * origin_h * sizeof(int16_t));
  }

  int deconv_idx = 0;
  for (int nn = 0; nn < num; ++nn) {
    for (int hh = 0; hh < origin_h; ++hh) {
      int hx = (hh % sub_conv_n);
      auto sub_t =
          (int16_t *)(args.split_conv_args[sub_conv_n - hx - 1]  // NOLINT
                          .output.address);
      int hi = (hh / sub_conv_n);
      if ((hh < omit_size) || (hh >= (origin_h - omit_size))) continue;
      int sidx = (nn * origin_h * align_origin_w + hi * align_origin_w +
                  omit_size * channel);
      fpga_copy((int16_t *)(args.output.address) + deconv_idx,    // NOLINT
                sub_t + sidx, sizeof(int16_t) * deconv_row_len);  // NOLINT
      deconv_idx += align_deconv_row_len;
    }
  }
  fpga_flush(args.output.address,
             num * align_deconv_row_len * deconv_h * sizeof(int16_t));
}

int ComputeFpgaDeconv(const struct DeconvArgs &args) {
#ifdef FPGA_PRINT_MODE
  DLOG << "=============ComputeFPGADeConv===========";
  DLOG << "   filter_num:" << args.filter_num
       << "   group_num:" << args.group_num << "omit_size:" << args.omit_size
       << "sub_output_width: " << args.sub_output_width
       << "sub_output_height: " << args.sub_output_height
       << "   sub_conv_num:" << args.sub_conv_num;
  DLOG << "args.output.address: " << args.output.address
       << "args.output.scale_address: " << args.output.scale_address;

#endif

  int sub_conv_num = args.sub_conv_num;

#ifdef COST_TIME_PRINT
  timeval start, end;
  long dif_sec, dif_usec;  // NOLINT
#endif

  for (int i = 0; i < sub_conv_num; i++) {
#ifdef COST_TIME_PRINT
    gettimeofday(&start, NULL);
#endif

    ComputeFpgaConv(args.split_conv_args[i]);
#ifdef COST_TIME_PRINT
    gettimeofday(&end, NULL);
    dif_sec = end.tv_sec - start.tv_sec;
    dif_usec = end.tv_usec - start.tv_usec;
    std::cout << "deconv basic_conv: " << i << " times:  "
              << "    cost time: " << (dif_sec * 1000000 + dif_usec) << "us"
              << std::endl;
#endif
  }

  if (sub_conv_num > 1) {
    float max_scale = -1.0f;
#ifdef COST_TIME_PRINT
    gettimeofday(&start, NULL);
#endif
    for (int i = 0; i < sub_conv_num; i++) {
      paddle_mobile::fpga::fpga_invalidate(
              args.split_conv_args[i].output.scale_address, 2 * sizeof(float));
      float ptr_scale = (args.split_conv_args[i].output.scale_address)[0];
      if (ptr_scale > max_scale) {
        args.output.scale_address[0] = ptr_scale;
        args.output.scale_address[1] =
                (args.split_conv_args[i].output.scale_address)[1];
      }
    }
  }

#ifdef COST_TIME_PRINT
    gettimeofday(&end, NULL);
    dif_sec = end.tv_sec - start.tv_sec;
    dif_usec = end.tv_usec - start.tv_usec;
    std::cout << "deconv scale  "
              << "    cost time: " << (dif_sec * 1000000 + dif_usec) << "us"
              << std::endl;
#endif

  return 0;
}  // ComputeFpgaDeconv

int ComputeFPGASplit(const struct SplitArgs &args) {
#ifdef FPGA_PRINT_MODE
  DLOG << "=============ComputeFpgaSplit===========";
  DLOG << "   Image_num: " << args.image_num
       << "   in_address:" << args.image_in
       << "   in_scale_address:" << args.scale_in;
  DLOG << "   image_height:" << args.height << "   image_width:" << args.width;
  for (int i = 0; i < args.image_num; i++) {
    DLOG << "   " << i << "th:        ";
    DLOG << "   channel_num:" << args.out_channel_nums[i]
         << "   image_address:" << args.images_out[i]
         << "   image_scale_address:" << args.scales_out[i];
  }
#endif
  image::split_image(args.image_in, args.scale_in, args.images_out,
                     args.scales_out, args.image_num, args.out_channel_nums,
                     args.height, args.width);
  return 0;
}  // ComputeFPGASplit

int ComputeDWConv(const struct DWconvArgs &args) {
#ifdef FPGA_PRINT_MODE
  DLOG << "=============ComputeDWConv===========";
  DLOG << "   mode:" << args.relu_enabled;
  DLOG << "   image_address:" << args.image.address
       << "   image_scale_address:" << args.image.scale_address
       << "   image_channels:" << args.image.channels
       << "   image_height:" << args.image.height
       << "   image_width:" << args.image.width
       << "   pad_height:" << args.image.pad_height
       << "   pad_width:" << args.image.pad_width;
  DLOG << "   filter_address:" << args.filter_address
       << "   bias_address:" << args.bias_address;
  DLOG << "   kernel_height:" << args.kernel.height
       << "   kernel_width:" << args.kernel.width
       << "   stride_h:" << args.kernel.stride_h
       << "   stride_w:" << args.kernel.stride_w;
  DLOG << "   out_address:" << args.output.address
       << "   out_scale_address:" << args.output.scale_address;
#endif

  return 0;
}
}  // namespace fpga
}  // namespace paddle_mobile
