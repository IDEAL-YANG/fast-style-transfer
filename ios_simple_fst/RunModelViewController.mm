// Copyright 2015 Google Inc. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//    http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#import "RunModelViewController.h"

#include <fstream>
#include <pthread.h>
#include <unistd.h>
#include <queue>
#include <sstream>
#include <string>

#include "google/protobuf/io/coded_stream.h"
#include "google/protobuf/io/zero_copy_stream_impl.h"
#include "google/protobuf/io/zero_copy_stream_impl_lite.h"
#include "google/protobuf/message_lite.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/framework/tensor.h"
#include "tensorflow/core/framework/types.pb.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/logging.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/public/session.h"

#include "ios_image_load.h"



//NSString* RunInferenceOnImage();
UIImage* RunInferenceOnImage();

namespace {
class IfstreamInputStream : public ::google::protobuf::io::CopyingInputStream {
 public:
  explicit IfstreamInputStream(const std::string& file_name)
      : ifs_(file_name.c_str(), std::ios::in | std::ios::binary) {}
  ~IfstreamInputStream() { ifs_.close(); }

  int Read(void* buffer, int size) {
    if (!ifs_) {
      return -1;
    }
    ifs_.read(static_cast<char*>(buffer), size);
    return ifs_.gcount();
  }

 private:
  std::ifstream ifs_;
};
}  // namespace


@interface UIImage (Resize)
- (UIImage*)scaleToSize:(CGSize)size;
@end


@implementation UIImage (Resizing)

- (UIImage*)scaleToSize:(CGSize)size {
    UIGraphicsBeginImageContext(size);
    
    CGContextRef context = UIGraphicsGetCurrentContext();
    CGContextTranslateCTM(context, 0.0, size.height);
    CGContextScaleCTM(context, 1.0, -1.0);
    
    CGContextDrawImage(context, CGRectMake(0.0f, 0.0f, size.width, size.height), self.CGImage);
    
    UIImage* scaledImage = UIGraphicsGetImageFromCurrentImageContext();
    
    UIGraphicsEndImageContext();
    
    return scaledImage;
}
@end


@interface RunModelViewController ()
@end

@implementation RunModelViewController {
}

- (IBAction)getUrl:(id)sender {
//  NSString* inference_result = RunInferenceOnImage();
//  self.urlContentTextView.text = inference_result;
    [self.ivStyled setImage:RunInferenceOnImage()];
}


+ (UIImage *) convertBitmapRGBA8ToUIImage:(unsigned char *) buffer
                                withWidth:(int) width
                               withHeight:(int) height {
    
    // added code
    char* rgba = (char*)malloc(width*height*4);
    for(int i=0; i < width*height; ++i) {
        rgba[4*i] = buffer[3*i];
        rgba[4*i+1] = buffer[3*i+1];
        rgba[4*i+2] = buffer[3*i+2];
        rgba[4*i+3] = 255;
    }
    //
    
    
    size_t bufferLength = width * height * 4;
    CGDataProviderRef provider = CGDataProviderCreateWithData(NULL, rgba, bufferLength, NULL);
    size_t bitsPerComponent = 8;
    size_t bitsPerPixel = 32;
    size_t bytesPerRow = 4 * width;
    
    CGColorSpaceRef colorSpaceRef = CGColorSpaceCreateDeviceRGB();
    if(colorSpaceRef == NULL) {
        NSLog(@"Error allocating color space");
        CGDataProviderRelease(provider);
        return nil;
    }
    
    CGBitmapInfo bitmapInfo = kCGBitmapByteOrderDefault | kCGImageAlphaPremultipliedLast;
    CGColorRenderingIntent renderingIntent = kCGRenderingIntentDefault;
    
    CGImageRef iref = CGImageCreate(width,
                                    height,
                                    bitsPerComponent,
                                    bitsPerPixel,
                                    bytesPerRow,
                                    colorSpaceRef,
                                    bitmapInfo,
                                    provider,   // data provider
                                    NULL,       // decode
                                    YES,            // should interpolate
                                    renderingIntent);
    
    uint32_t* pixels = (uint32_t*)malloc(bufferLength);
    
    if(pixels == NULL) {
        NSLog(@"Error: Memory not allocated for bitmap");
        CGDataProviderRelease(provider);
        CGColorSpaceRelease(colorSpaceRef);
        CGImageRelease(iref);
        return nil;
    }
    
    CGContextRef context = CGBitmapContextCreate(pixels,
                                                 width,
                                                 height,
                                                 bitsPerComponent,
                                                 bytesPerRow,
                                                 colorSpaceRef,
                                                 bitmapInfo);
    
    if(context == NULL) {
        NSLog(@"Error context not created");
        free(pixels);
    }
    
    UIImage *image = nil;
    if(context) {
        
        CGContextDrawImage(context, CGRectMake(0.0f, 0.0f, width, height), iref);
        
        CGImageRef imageRef = CGBitmapContextCreateImage(context);
        
        // Support both iPad 3.2 and iPhone 4 Retina displays with the correct scale
        if([UIImage respondsToSelector:@selector(imageWithCGImage:scale:orientation:)]) {
            float scale = [[UIScreen mainScreen] scale];
            image = [UIImage imageWithCGImage:imageRef scale:scale orientation:UIImageOrientationUp];
        } else {
            image = [UIImage imageWithCGImage:imageRef];
        }
        
        CGImageRelease(imageRef);
        CGContextRelease(context);
    }
    
    CGColorSpaceRelease(colorSpaceRef);
    CGImageRelease(iref);
    CGDataProviderRelease(provider);
    
    if(pixels) {
        free(pixels);
    }   
    return image;
}

@end


static UIImage* tensorToBuffer(
                    const Eigen::TensorMap<Eigen::Tensor<float, 1, Eigen::RowMajor>,
                    Eigen::Aligned>& outputTensor, int image_width, int image_height) {
    const int count = outputTensor.size();
    unsigned char* buffer = (unsigned char*)malloc(count);
    for (int i = 0; i < count; ++i) {
        const float value = outputTensor(i);
        int n;
        if (value < 0) n = 0;
        else if (value > 255) n = 255;
        else n = (int)value;
        buffer[i] = n;
    }
    
    UIImage *img = [RunModelViewController convertBitmapRGBA8ToUIImage:buffer withWidth: 420 withHeight: 560];
    UIImage *imgScaled = [img scaleToSize:CGSizeMake(image_width, image_height)];
    
    NSArray *paths = NSSearchPathForDirectoriesInDomains(NSDocumentDirectory, NSUserDomainMask, YES);
    NSString *filePath = [[paths objectAtIndex:0] stringByAppendingPathComponent:@"fns.jpg"];
    
    // Save image.
    [UIImageJPEGRepresentation(imgScaled, 1.0) writeToFile:filePath atomically:YES];
    
    NSLog(@"filePath=%@", filePath);
    
    return imgScaled;
    
}


bool PortableReadFileToProto(const std::string& file_name,
                             ::google::protobuf::MessageLite* proto) {
  ::google::protobuf::io::CopyingInputStreamAdaptor stream(
      new IfstreamInputStream(file_name));
  stream.SetOwnsCopyingStream(true);
  // TODO(jiayq): the following coded stream is for debugging purposes to allow
  // one to parse arbitrarily large messages for MessageLite. One most likely
  // doesn't want to put protobufs larger than 64MB on Android, so we should
  // eventually remove this and quit loud when a large protobuf is passed in.
  ::google::protobuf::io::CodedInputStream coded_stream(&stream);
  // Total bytes hard limit / warning limit are set to 1GB and 512MB
  // respectively. 
  coded_stream.SetTotalBytesLimit(1024LL << 20, 512LL << 20);
  return proto->ParseFromCodedStream(&coded_stream);
}

NSString* FilePathForResourceName(NSString* name, NSString* extension) {
  NSString* file_path = [[NSBundle mainBundle] pathForResource:name ofType:extension];
  if (file_path == NULL) {
    LOG(FATAL) << "Couldn't find '" << [name UTF8String] << "."
	       << [extension UTF8String] << "' in bundle.";
  }
  return file_path;
}

//NSString* RunInferenceOnImage() {
UIImage* RunInferenceOnImage() {
  tensorflow::SessionOptions options;

  tensorflow::Session* session_pointer = nullptr;
  tensorflow::Status session_status = tensorflow::NewSession(options, &session_pointer);
  if (!session_status.ok()) {
    std::string status_string = session_status.ToString();
      return NULL; //[NSString stringWithFormat: @"Session create failed - %s",status_string.c_str()];
  }
  std::unique_ptr<tensorflow::Session> session(session_pointer);
  LOG(INFO) << "Session created.";

  tensorflow::GraphDef tensorflow_graph;
  LOG(INFO) << "Graph created.";

  NSString* network_path = FilePathForResourceName(@"fst_frozen_quantized", @"pb");
  PortableReadFileToProto([network_path UTF8String], &tensorflow_graph);

  LOG(INFO) << "Creating session.";
  tensorflow::Status s = session->Create(tensorflow_graph);
  if (!s.ok()) {
    LOG(ERROR) << "Could not create TensorFlow Graph: " << s;
      return NULL; //@"";
  }

    static int rotate = 0;
    NSString* image_path = FilePathForResourceName(@"ww2", @"jpg");
    if (rotate == 1) image_path = FilePathForResourceName(@"IMG_0777", @"jpg");
    else if (rotate == 2) image_path = FilePathForResourceName(@"IMG_5886", @"png");
    else image_path = FilePathForResourceName(@"ww2", @"jpg");
    if (++rotate > 2) rotate = 0;

    int image_width;
  int image_height;
  int image_channels;
  std::vector<tensorflow::uint8> image_data = LoadImageFromFile(
	[image_path UTF8String], &image_width, &image_height, &image_channels);
    
    const int wanted_width = 420;
    const int wanted_height =560;
  const int wanted_channels = 3;
  const float input_mean = 128.0f;
  const float input_std = 128.0f;

  assert(image_channels >= wanted_channels);
  tensorflow::Tensor image_tensor(
      tensorflow::DT_FLOAT,
      tensorflow::TensorShape({
          wanted_height, wanted_width, wanted_channels}));
  auto image_tensor_mapped = image_tensor.tensor<float, 3>();
  tensorflow::uint8* in = image_data.data();
  tensorflow::uint8* in_end = (in + (image_height * image_width * image_channels));
  float* out = image_tensor_mapped.data();
  for (int y = 0; y < wanted_height; ++y) {
    const int in_y = (y * image_height) / wanted_height;
    tensorflow::uint8* in_row = in + (in_y * image_width * image_channels);
    float* out_row = out + (y * wanted_width * wanted_channels);
    for (int x = 0; x < wanted_width; ++x) {
      const int in_x = (x * image_width) / wanted_width;
      tensorflow::uint8* in_pixel = in_row + (in_x * image_channels);
      float* out_pixel = out_row + (x * wanted_channels);
      for (int c = 0; c < wanted_channels; ++c) {
        out_pixel[c] = (in_pixel[c] - input_mean) / input_std;
      }
    }
  }


  std::string input_layer = "img_placeholder";
  std::string output_layer = "preds";
    
  std::vector<tensorflow::Tensor> outputs;
  tensorflow::Status run_status = session->Run({{input_layer, image_tensor}},
				               {output_layer}, {}, &outputs);
  if (!run_status.ok()) {
    LOG(ERROR) << "Running model failed: " << run_status;
      return NULL;
  }
  tensorflow::string status_string = run_status.ToString();
  tensorflow::Tensor* output = &outputs[0];
  const int kNumResults = 5;
  const float kThreshold = 0.1f;
  std::vector<std::pair<float, int> > top_results;
    
  UIImage *imgScaled = tensorToBuffer(output->flat<float>(), image_width, image_height);
  return imgScaled;
}


