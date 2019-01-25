// /* Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.

// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at

//     http://www.apache.org/licenses/LICENSE-2.0

// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License. */
// #include <fstream>
// #include <iomanip>
// #include <iostream>
// #include "../test_include.h"
// #include <typeinfo>
// #include <typeindex>

// // #include <opencv2/dnn.hpp>
// #include <opencv2/imgproc.hpp>
// #include <opencv2/highgui.hpp>
// // #include <opencv2/core/utils/trace.hpp>
// using namespace cv;

// #ifdef PADDLE_MOBILE_FPGA_V1
// // #include "fpga/V1/half.h"
// #include "fpga/V1/api.h"
// #endif
// #ifdef PADDLE_MOBILE_FPGA_V2
// #include "fpga/V2/api.h"
// #endif

// #ifdef PADDLE_MOBILE_FPGA_KD
// #include "fpga-kd/api.h"
// #endif

// using namespace std;

// // cv::Mat sample_float;

// // void readImage(std::string filename, float* buffer) {
// //   Mat img = imread(filename);
// //   if (img.empty()) {
// //       std::cerr << "Can't read image from the file: " << filename << std::endl;
// //       exit(-1);
// //   }
  
// //   // Mat img2;
// //   // resize(img, img2, Size(300,300));

// //   img.convertTo(sample_float, CV_32FC3);

// //   int index = 0;
// //   for (int row = 0; row < sample_float.rows; ++row) {
// //     float* ptr = (float*)sample_float.ptr(row);
// //     for (int col = 0; col < sample_float.cols; col++) {
// //         float* uc_pixel = ptr;
// //         // uc_pixel[0] -= 102;
// //         // uc_pixel[1] -= 117;
// //         // uc_pixel[1] -= 124;
// //         float r = uc_pixel[0];
// //         float g = uc_pixel[1];
// //         float b = uc_pixel[2];     

// //         buffer[index] = b - 104;
// //         buffer[index + 1] = g - 117;
// //         buffer[index + 2] = r - 124;

// //         // sum += a + b + c;
// //         ptr += 3;
// //         // DLOG << "r:" << r << " g:" << g << " b:" << b;
// //         index += 3;
// //     }
// //   }
// //   // return sample_float;
// // }

// // void drawRect(Mat& mat, float* data,int len) {
// //   for (int i = 0;i < len; i++) {

// //     float index = data[0];
// //     float score = data[1];

// //     if (score < 0.5f) {
// //       continue;
// //     }

// //     float x0 = data[2] * 300;
// //     float y0 = data[3] * 300;
// //     float x1 = data[4] * 300;
// //     float y1 = data[5] * 300;
// //     int width = 10;
// //     int height = 20;
// //     // our rectangle...
// //     // cv::Rect rect(x0, y0, width, height);
// //     // and its top left corner...
// //     cv::Point pt1(x0, y0);
// //     // and its bottom right corner.
// //     cv::Point pt2(x1, y1);
// //     // cv::Point pt2(x + width, y + height);
// //     // These two calls...
// //     cv::rectangle(mat, pt1, pt2, cv::Scalar(0, 0, 255));

// //     DLOG << "score::" << score;
// //     data += 6;
// //   }
// //   imwrite( "result.jpg", mat );
// //   // Mat inputBlob = blobFromImage(img2, 0.007843, Size(300,300), Scalar(104.0, 117, 124), false);
// // }

// void readStream(std::string filename, float *buf) {
//   std::ifstream in;
//   in.open(filename, std::ios::in);
//   if (!in.is_open()) {
//     std::cout << "open File Failed." << std::endl;
//     return;
//   }
//   string strOne;
//   int i = 0;
//   while (!in.eof()) {
//     in >> buf[i];
//     i++;
//   }
//   in.close();
// }

// void dump(std::string filename, const Tensor input_tensor,int pe_index) {
//   std::cout << input_tensor.type().name() << std::endl;
//   DLOG << "result:::" << input_tensor.dims();
//   if (input_tensor.type() == typeid(float)) {
//     auto dataptr = input_tensor.data<float>();
//     std::ofstream out(filename.c_str());

//     int16_t* dd = (int16_t*)dataptr;
//     int* int_data = (int*)dataptr;
//     float* float_data = (float*)dataptr;
    
//     float result = 0;
//     int num = 1;
//     int height = 1;
//     int width = 1;
//     int channel = 1;
//     if (input_tensor.dims().size() == 4) {
//       num = input_tensor.dims()[0];
//       channel = input_tensor.dims()[1];
//       height = input_tensor.dims()[2];
//       width = input_tensor.dims()[3];
//     }

//     if (input_tensor.dims().size() == 3) {
//       num = input_tensor.dims()[0];
//       channel = input_tensor.dims()[2];
//       height = input_tensor.dims()[1];
//     }
//     if (input_tensor.dims().size() == 2) {
//       channel = input_tensor.dims()[1];
//       height = input_tensor.dims()[0];
//     }
//     if (input_tensor.dims().size() == 1) {
//       num = input_tensor.dims()[0];
//     }

//     int index = 0;
//     int cw = channel * width;
//     int align_cw = fpga::align_to_x(cw, 16);
//     int remainder = align_cw - cw;

//     DLOG << "data_aligned:" << input_tensor.data_aligned();

//     for (int n = 0; n < num; n++){
//       for (int h = 0; h < height; h++) {
//         for (int i = 0; i < cw; i++) {
//           // out << index << std::endl;
//           if (input_tensor.type() == typeid(float)) {
//             if (pe_index >= 96) {
//               result = float_data[index];
//             }else{
//               result = paddle_mobile::fpga::fp16_2_fp32(dd[index]);
//             }
            
//             out << result << std::endl;
//           }else{
//             out << int_data[index] << std::endl;
//           }

//           index++;
//         }
//         if (input_tensor.data_aligned()) {
//           for (int i = 0; i < remainder; i++) {
//             index++;
//           }
//         }
//       }
//     }
//     out.close();
//   } else {
//     auto dataptr = input_tensor.data<int>();
//     std::ofstream out(filename.c_str());

//     int16_t* dd = (int16_t*)dataptr;
//     int result = 0;
//     for (int i = 0; i < input_tensor.numel(); ++i) {
//       out << result << std::endl;
//     }
//     out.close();
//   }
// }

// void run_entire(paddle_mobile::PaddleMobile<paddle_mobile::FPGA>& paddle_mobile) {
//   paddle_mobile.Predict_To(-1);

//   for (int i = 0; i < 90; i++) {
//     // paddle_mobile.Predict_From_To(0,1);
//     int out_num = paddle_mobile.OutputsNum(i);
//     for (int n = 0;n < out_num; n++) {
//       auto tensor_ptr = paddle_mobile.FetchResult(i, n);
//       std::string saveName = "output/" + std::to_string(i) + 
//       + "_" + std::to_string(n) + ".txt";
//       if ((*tensor_ptr).type() == typeid(float)) {
//         paddle_mobile::fpga::fpga_invalidate((*tensor_ptr).data<float>(),
//                                            tensor_ptr->numel() * sizeof(half));
//       }else {
//         paddle_mobile::fpga::fpga_invalidate((*tensor_ptr).data<int>(),
//                                            tensor_ptr->numel() * sizeof(int));
//       }
//       dump(saveName, (*tensor_ptr),i);
//     }
//   }
// }


// static const char *g_ssd = "../models/ssd";
// int main() {
//   paddle_mobile::fpga::open_device();

//   paddle_mobile::PaddleMobile<paddle_mobile::FPGA> paddle_mobile;
//   std::string model = std::string(g_ssd) + "/model";
//   std::string params = std::string(g_ssd) + "/params";
//   if (paddle_mobile.Load(model, params, true)) {
//     Tensor input_tensor;
//     SetupTensor<float>(&input_tensor, {1, 3, 300, 300}, static_cast<float>(2),
//                        static_cast<float>(2));
//     float* data = input_tensor.mutable_data<float>({1, 3, 300, 300});
//     // readImage("2.jpg", data);
//     for (int i = 0; i < 3 * 300 * 300; i++) {
//       data[i] = 1.0f;
//     }

//     paddle_mobile.FeedData(input_tensor);
    
//     // run_entire(paddle_mobile);
//     auto time3 = time();
//     for (int i = 0;i < 2;i++){
      
//       run_entire(paddle_mobile);
//     }
    
//     auto time4 = time();
//     std::cout << "predict cost: " << time_diff(time3, time4) << "ms\n";

    
//     auto result_ptr = paddle_mobile.FetchResult();
//     float* result_data = result_ptr->data<float>();

//     // drawRect(sample_float, result_data,result_ptr->dims()[0]);

//     // for (int i = 0;i < result_ptr->numel(); i++) {
//     //   DLOG << result_data[i] ;
//     // }
//     // run_single_step(paddle_mobile);
//   }
//   return 0;
// }



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
#include <fstream>
#include <iomanip>
#include <iostream>
#include "../test_include.h"
#include <typeinfo>
#include <typeindex>

// #include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
// #include <opencv2/core/utils/trace.hpp>
using namespace cv;

#ifdef PADDLE_MOBILE_FPGA_V1
// #include "fpga/V1/half.h"
#include "fpga/V1/api.h"
#endif
#ifdef PADDLE_MOBILE_FPGA_V2
#include "fpga/V2/api.h"
#endif

using namespace std;

cv::Mat sample_float;

void readImage(std::string filename, float* buffer) {
  Mat img = imread(filename);
  if (img.empty()) {
      std::cerr << "Can't read image from the file: " << filename << std::endl;
      exit(-1);
  }
  
  // Mat img2;
  // resize(img, img2, Size(300,300));

  img.convertTo(sample_float, CV_32FC3);

  int index = 0;
  for (int row = 0; row < sample_float.rows; ++row) {
    float* ptr = (float*)sample_float.ptr(row);
    for (int col = 0; col < sample_float.cols; col++) {
        float* uc_pixel = ptr;
        // uc_pixel[0] -= 102;
        // uc_pixel[1] -= 117;
        // uc_pixel[1] -= 124;
        float r = uc_pixel[0];
        float g = uc_pixel[1];
        float b = uc_pixel[2];     

        buffer[index] = b - 104;
        buffer[index + 1] = g - 117;
        buffer[index + 2] = r - 124;

        // sum += a + b + c;
        ptr += 3;
        // DLOG << "r:" << r << " g:" << g << " b:" << b;
        index += 3;
    }
  }
  // return sample_float;
}

void drawRect(Mat& mat, float* data,int len) {
  for (int i = 0;i < len; i++) {

    float index = data[0];
    float score = data[1];

    if (score < 0.5f) {
      continue;
    }

    float x0 = data[2] * 300;
    float y0 = data[3] * 300;
    float x1 = data[4] * 300;
    float y1 = data[5] * 300;
    int width = 10;
    int height = 20;
    // our rectangle...
    // cv::Rect rect(x0, y0, width, height);
    // and its top left corner...
    cv::Point pt1(x0, y0);
    // and its bottom right corner.
    cv::Point pt2(x1, y1);
    // cv::Point pt2(x + width, y + height);
    // These two calls...
    cv::rectangle(mat, pt1, pt2, cv::Scalar(0, 0, 255));

    DLOG << "score::" << score;
    data += 6;
  }
  imwrite( "result.jpg", mat );
  // Mat inputBlob = blobFromImage(img2, 0.007843, Size(300,300), Scalar(104.0, 117, 124), false);
}

void readStream(std::string filename, float *buf) {
  std::ifstream in;
  in.open(filename, std::ios::in);
  if (!in.is_open()) {
    std::cout << "open File Failed." << std::endl;
    return;
  }
  string strOne;
  int i = 0;
  while (!in.eof()) {
    in >> buf[i];
    i++;
  }
  in.close();
}

void dump(std::string filename, const Tensor input_tensor,int pe_index) {
  std::cout << input_tensor.type().name() << std::endl;
  DLOG << "result:::" << input_tensor.dims();
  if (input_tensor.type() == typeid(float)) {
    auto dataptr = input_tensor.data<float>();
    std::ofstream out(filename.c_str());

    int16_t* dd = (int16_t*)dataptr;
    int* int_data = (int*)dataptr;
    float* float_data = (float*)dataptr;
    
    float result = 0;
    int num = 1;
    int height = 1;
    int width = 1;
    int channel = 1;
    if (input_tensor.dims().size() == 4) {
      num = input_tensor.dims()[0];
      channel = input_tensor.dims()[1];
      height = input_tensor.dims()[2];
      width = input_tensor.dims()[3];
    }

    if (input_tensor.dims().size() == 3) {
      num = input_tensor.dims()[0];
      channel = input_tensor.dims()[2];
      height = input_tensor.dims()[1];
    }
    if (input_tensor.dims().size() == 2) {
      channel = input_tensor.dims()[1];
      height = input_tensor.dims()[0];
    }
    if (input_tensor.dims().size() == 1) {
      num = input_tensor.dims()[0];
    }

    int index = 0;
    int cw = channel * width;
    int align_cw = fpga::align_to_x(cw, 16);
    int remainder = align_cw - cw;

    DLOG << "data_aligned:" << input_tensor.data_aligned();

    for (int n = 0; n < num; n++){
      for (int h = 0; h < height; h++) {
        for (int i = 0; i < cw; i++) {
          // out << index << std::endl;
          if (input_tensor.type() == typeid(float)) {
            if (pe_index >= 96) {
              result = float_data[index];
            }else{
              result = paddle_mobile::fpga::fp16_2_fp32(dd[index]);
            }
            
            out << result << std::endl;
          }else{
            out << int_data[index] << std::endl;
          }

          index++;
        }
        if (input_tensor.data_aligned()) {
          for (int i = 0; i < remainder; i++) {
            index++;
          }
        }
      }
    }
    out.close();
  } else {
    auto dataptr = input_tensor.data<int>();
    std::ofstream out(filename.c_str());

    int16_t* dd = (int16_t*)dataptr;
    int result = 0;
    for (int i = 0; i < input_tensor.numel(); ++i) {
      out << result << std::endl;
    }
    out.close();
  }
}

void run_entire(paddle_mobile::PaddleMobile<paddle_mobile::FPGA>& paddle_mobile) {
  paddle_mobile.Predict_To(-1);

  // for (int i = 0; i < 104; i++) {
  //   // paddle_mobile.Predict_From_To(0,1);
  //   int out_num = paddle_mobile.OutputsNum(i);
  //   for (int n = 0;n < out_num; n++) {
  //     auto tensor_ptr = paddle_mobile.FetchResult(i, n);
  //     std::string saveName = "output/" + std::to_string(i) + 
  //     + "_" + std::to_string(n) + ".txt";
  //     if ((*tensor_ptr).type() == typeid(float)) {
  //       paddle_mobile::fpga::fpga_invalidate((*tensor_ptr).data<float>(),
  //                                          tensor_ptr->numel() * sizeof(half));
  //     }else {
  //       paddle_mobile::fpga::fpga_invalidate((*tensor_ptr).data<int>(),
  //                                          tensor_ptr->numel() * sizeof(int));
  //     }

  //     // if (!tensor_ptr->data_aligned()) {
  //     //   dump(saveName, (*tensor_ptr));
  //     // }
  //     dump(saveName, (*tensor_ptr),i);
  //     // DLOG << "aligned:" << tensor_ptr->data_aligned();

  //   }
  // }
}


static const char *g_ssd = "../models/ssd";
int main() {
  paddle_mobile::fpga::open_device();

  paddle_mobile::PaddleMobile<paddle_mobile::FPGA> paddle_mobile;
  std::string model = std::string(g_ssd) + "/model";
  std::string params = std::string(g_ssd) + "/params";
  if (paddle_mobile.Load(model, params, true)) {
    Tensor input_tensor;
    SetupTensor<float>(&input_tensor, {1, 3, 300, 300}, static_cast<float>(2),
                       static_cast<float>(2));
    float* data = input_tensor.mutable_data<float>({1, 3, 300, 300});
    readImage("2.jpg", data);
    // for (int i = 0; i < 3 * 300 * 300; i++) {
    //   data[i] = 1.0f;
    // }

    paddle_mobile.FeedData(input_tensor);
    
    // run_entire(paddle_mobile);
    auto time3 = time();
    for (int i = 0;i < 10;i++){
      
      run_entire(paddle_mobile);
    }
    
    auto time4 = time();
    std::cout << "predict cost: " << time_diff(time3, time4) / 10 << "ms\n";

    
    auto result_ptr = paddle_mobile.FetchResult();
    float* result_data = result_ptr->data<float>();

    drawRect(sample_float, result_data,result_ptr->dims()[0]);

    // for (int i = 0;i < result_ptr->numel(); i++) {
    //   DLOG << result_data[i] ;
    // }
    // run_single_step(paddle_mobile);
  }
  return 0;
}
