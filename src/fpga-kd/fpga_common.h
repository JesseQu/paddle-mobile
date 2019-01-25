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


#ifndef PADDLE_MOBILE_SRC_FPGA_KD_ZYNQMP_API_H
#define PADDLE_MOBILE_SRC_FPGA_KD_ZYNQMP_API_H

#include <stdint.h>
#include <cstddef>
#include <iostream>
#include <limits>

#include "common/types.h"

namespace paddle_mobile {
namespace fpga {

    #define IMAGE_ALIGNMENT 16           // Aligned to 16
    #define FILTER_NUM_ALIGNMENT 32      // Filter number aligned to 32
    #define FILTER_ELEMENT_ALIGNMENT 16  // Filter element number aligned to 16
    #define BS_NUM_ALIGNMENT 8
    #define BIAS_NUM_ALIGNMENT 16


    enum DataType {
        DATA_TYPE_FP32 = 1,
        DATA_TYPE_FP16 = 0,
    };

    enum LayoutType {
        LAYOUT_CHW = 1,
        LAYOUT_HWC = 0,
    };

    enum ActivationType {
        NONE = 0,
        LEAKYRELU = 1,
        SIGMOID = 2,
        TANH = 3,
    };

    struct VersionArgs {
        void* buffer;
    };

    struct MemoryCopyArgs {
        void* src;
        void* dest;
        size_t size;
    };

    struct ActivationArgs {
        enum ActivationType activation_type;
        int16_t leaky_relu_negative_slope;
    };

/**
Conv and Pooling kernel
*/
    struct KernelArgs {
        uint32_t width;
        uint32_t height;
        uint32_t stride_w;
        uint32_t stride_h;
    };

    struct ImageInputArgs {
        void* address;         // input featuremap virtual address
        void* scale_address;  // input scale address;
        uint32_t channels;
        uint32_t width;  // featuremap width
        uint32_t height;
        uint32_t pad_width;  // padding width;
        uint32_t pad_height;
    };

    struct ImageOutputArgs {
        void* address;         // output result address;
        float* scale_address;  // output scale address;
    };

    struct ConvArgs {
        bool relu_enabled;
        void* sb_address;  // scale and bias are interlaced;
        void* filter_address;
        void* filter_scale_address;
        uint32_t filter_num;
        uint32_t group_num;

        struct KernelArgs kernel;
        struct ImageInputArgs image;  // input image;
        struct ImageOutputArgs output;
    };

    struct PoolingArgs {
        uint16_t mode;
        uint16_t kernel_reciprocal;
        struct KernelArgs kernel;
        struct ImageInputArgs image;  // input image;
        struct ImageOutputArgs output;
        uint16_t out_width;
        uint16_t out_height;
    };

// elementwise add arguments
    struct EWAddArgs {
        bool relu_enabled;

        uint32_t const0;  // output0 = const0 x input0 + const1 x input1;
        uint32_t const1;
        struct ImageInputArgs image0;
        struct ImageInputArgs image1;
        struct ImageOutputArgs output;
    };

    struct BypassArgs {
        enum DataType input_data_type;
        enum DataType output_data_type;
        enum LayoutType input_layout_type;
        enum LayoutType output_layout_type;
        struct ImageInputArgs image;
        struct ImageOutputArgs output;
    };

    struct FpgaRegWriteArgs {
        uint64_t address;  //
        uint64_t value;
    };

    struct FpgaRegReadArgs {
        uint64_t address;
        uint64_t value;
    };

    struct MemoryCacheArgs {
        void* address;
        size_t size;
    };

    struct FpgaResetArgs {
    };

    struct PowerArgs {
        uint16_t                shift;
        uint16_t                scale;
        uint16_t                power;
    };

    struct SplitArgs {
        uint32_t image_num;
        int16_t* image_in;
        float* scale_in;
        void** images_out;
        float** scales_out;
        uint32_t* out_channel_nums;
        uint32_t height;
        uint32_t width;
    };

    struct InplaceArgs {
        bool                    relu_enable;
        bool                    power_enable;
    };

    struct ScaleArgs {
        void*                   scale_address;
        void*                   bias_address;
        uint32_t                wc_alignment;
        uint32_t                channel_alignment;

        struct ImageInputArgs   image;
        struct ImageOutputArgs  output;
    };

    struct DeconvArgs {
        uint32_t sub_conv_num;
        uint32_t group_num;
        uint32_t filter_num;
        uint32_t omit_size;
        uint32_t sub_output_width;
        uint32_t sub_output_height;
        struct ImageOutputArgs output;
        struct SplitConvArgs* split_conv_args;
    };

    struct DWconvArgs {
        bool relu_enabled;
        void* bias_address;
        void* filter_address;
        struct KernelArgs kernel;
        struct ImageInputArgs image;
        struct ImageOutputArgs output;
    };

#define IOCTL_FPGA_MAGIC 'FPGA'

#define IOCTL_VERSION _IOW(IOCTL_FPGA_MAGIC, 01, struct VersionArgs)

// #define IOCTL_SEPARATOR_0 10
const int IOCTL_SEPARATOR_0 = 0;

#define IOCTL_MEM_COPY _IOW(IOCTL_FPGA_MAGIC, 11, struct MemoryCopyArgs)
#define IOCTL_MEMCACHE_INVAL _IOW(IOCTL_FPGA_MAGIC, 12, struct MemoryCacheArgs)
#define IOCTL_MEMCACHE_FLUSH _IOW(IOCTL_FPGA_MAGIC, 13, struct MemoryCacheArgs)

// #define IOCTL_SEPARATOR_1 20
const int IOCTL_SEPARATOR_1 = 20;

#define IOCTL_CONFIG_CONV       _IOW(IOCTL_FPGA_MAGIC, 21, struct ConvArgs)
#define IOCTL_CONFIG_POOLING    _IOW(IOCTL_FPGA_MAGIC, 22, struct PoolingArgs)
#define IOCTL_CONFIG_EW         _IOW(IOCTL_FPGA_MAGIC, 23, struct EWAddArgs)
#define IOCTL_CONFIG_BYPASS     _IOW(IOCTL_FPGA_MAGIC, 24, struct BypassArgs)
#define IOCTL_CONFIG_POWER      _IOW(IOCTL_FPGA_MAGIC, 25, struct PowerArgs)
#define IOCTL_CONFIG_SCALE      _IOW(IOCTL_FPGA_MAGIC, 26, struct ScaleArgs)
#define IOCTL_CONFIG_INPLACE    _IOW(IOCTL_FPGA_MAGIC, 27, struct InplaceArgs)
#define IOCTL_FPGA_REG_READ     _IOW(IOCTL_FPGA_MAGIC, 50, struct FpgaRegReadArgs)
#define IOCTL_FPGA_REG_WRITE    _IOW(IOCTL_FPGA_MAGIC, 51, struct FpgaRegWriteArgs)
#define IOCTL_FPGA_RESET        _IOW(IOCTL_FPGA_MAGIC, 52, struct FpgaResetArgs)

        enum FPGA_ERR_TYPE {
            ERR_IOCTL_CMD = -1,
            ERR_TIMEOUT = -2,
            ERR_COMPLETION_TIMEOUT = -3,
            ERR_INVALID_FPGA_ADDR = -4,
            ERR_NOMEM = -5,
            ERR_NO_RESERVE_MEM = -6,
            ERR_COPY_FROM_USER = -7,
            ERR_COPY_TO_USER = -8,
            ERR_DEL_TIMER = -9,
            ERR_ENABLE_MSI = -10,
            ERR_REGISTER_IRQ = -11,
            ERR_PCIE_REGISTER = -12,
            ERR_PCIE_PROBE = -13,
            ERR_REGISTER_BLOCK = -14,
            ERR_ALLOC_GENDISK = -15,
            ERR_INIT_QUEUE = -16,
            ERR_WAIT = -17,
            ERR_ECC_ERROR = -31,
            ERR_FPGA_FAIL_STOP = -64,
            ERR_FPGA_DEBUG_STOP = -113,
            DEV_TMP_UNAVAILABLE = -128
        };

//============================== API =============================

struct ConcatArgs {
  uint32_t image_num;
  half** images_in;
  float** scales_in;
  void* image_out;
  float* scale_out;
  uint32_t* channel_num;
  uint32_t height;
  uint32_t width;
};

struct SplitConvArgs {
  uint32_t split_num;
  uint32_t group_num;
  uint32_t filter_num;
  struct ImageOutputArgs output;
  struct ConvArgs* conv_arg;
  struct ConcatArgs concat_arg;
};

struct GroupConvArgs {
  uint32_t group_num;
  uint32_t filter_num;
  struct ImageOutputArgs output;
  struct SplitConvArgs* conv_args;
  struct ConcatArgs concat_arg;
};

        
    static inline int align_to_x(int num, int x) { return (num + x - 1) / x * x; }
    int open_device();
    void close_device();

    void reset_device();

    void* fpga_malloc(size_t size);
    void fpga_free(void* ptr);
    size_t fpga_get_memory_size(void* ptr);
    size_t fpga_get_memory_size_max();
    size_t fpga_diagnose_memory(int detailed);
    void fpga_copy(void* dst, const void* src, int size);

    int fpga_flush(void* address, size_t size);
    int fpga_invalidate(void* address, size_t size);

    int perform_bypass(const struct BypassArgs& args);
    int compute_fpga_conv_basic(const struct ConvArgs& args);
    int compute_fpga_conv(const struct SplitConvArgs& args);
    int compute_fpga_pool(const struct PoolingArgs& args);
    int compute_fpga_ewadd(const struct EWAddArgs& args);
    int compute_fpga_scale(const struct ScaleArgs& args);
    int compute_fpga_concat(const struct ConcatArgs& args);
    int config_power(const struct PowerArgs& args);

    // int config_relu(const struct ReluArgs& args);

    int config_inplace(const struct InplaceArgs& args);

    int flush_cache(void* addr, int size);
    int invalidate_cache(void* addr, int size);

//    void fill_conv_arg(WrapperConvArgs& args,  saber::Tensor<ZYNQMP>* input, saber::Tensor* output,
//                           char* weight, saber::Shape weight_shape, float* weight_scale, ConvParam& param, float* bs_data)

    uint64_t vaddr_to_paddr(void *address);

    int16_t fp32_2_fp16(float fp32_num);
    float fp16_2_fp32(int16_t fp16_num);


} // fpga
} // paddle_mobile;

#endif // PADDLE_MOBILE_SRC_FPGA_KD_ZYNQMP_API_H


