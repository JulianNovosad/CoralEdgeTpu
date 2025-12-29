#include "image_processor.h"
#include "util_logging.h"
#include "application.h"
#include <chrono>
#include <libcamera/formats.h>

int libcamera_pixel_format_to_opencv_type(const libcamera::PixelFormat& format) {
    if (format.fourcc() == libcamera::formats::BGRA8888.fourcc()) return CV_8UC4;
    if (format.fourcc() == libcamera::formats::BGR888.fourcc()) return CV_8UC3;
    if (format.fourcc() == libcamera::formats::RGBA8888.fourcc()) return CV_8UC4;
    if (format.fourcc() == libcamera::formats::RGB888.fourcc()) return CV_8UC3;
    if (format.fourcc() == libcamera::formats::YUYV.fourcc()) return CV_8UC2;
    return -1;
}

ImageProcessor::ImageProcessor(ImageQueue& input_queue, ImageQueue& output_queue,
                               TripleBuffer<DetectionResults>* detection_buffer,
                               std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                               libcamera::PixelFormat input_pixel_format,
                               int output_width, int output_height)
    : input_queue_(input_queue), 
      output_queue_(output_queue), 
      detection_buffer_ptr_(detection_buffer),
      buffer_pool_(buffer_pool),
      input_pixel_format_(input_pixel_format),
      output_width_(output_width), 
      output_height_(output_height),
      skip_factor_(1),
      frame_counter_(0),
      running_(false),
      avg_queue_pop_time_us_(0),
      avg_preprocess_time_us_(0),
      app_ref_(nullptr) {
}

ImageProcessor::ImageProcessor(ImageQueue& input_queue, ImageQueue& output_queue,
                               std::shared_ptr<BufferPool<uint8_t>> buffer_pool,
                               libcamera::PixelFormat input_pixel_format,
                               int output_width, int output_height)
    : input_queue_(input_queue), 
      output_queue_(output_queue), 
      detection_buffer_ptr_(nullptr),
      buffer_pool_(buffer_pool),
      input_pixel_format_(input_pixel_format),
      output_width_(output_width), 
      output_height_(output_height),
      skip_factor_(1),
      frame_counter_(0),
      running_(false),
      avg_queue_pop_time_us_(0),
      avg_preprocess_time_us_(0),
      app_ref_(nullptr) {
}

ImageProcessor::~ImageProcessor() {
    stop();
}

bool ImageProcessor::start() {
    if (!running_.exchange(true)) {
        worker_thread_ = std::thread(&ImageProcessor::worker_thread_func, this);
        return true;
    }
    return false;
}

void ImageProcessor::stop() {
    if (running_.exchange(false)) {
        if (worker_thread_.joinable()) {
            worker_thread_.join();
        }
    }
}

bool ImageProcessor::is_running() const {
    return running_.load();
}

void ImageProcessor::worker_thread_func() {
    set_thread_name("ImageProcessor");

    int opencv_input_type = libcamera_pixel_format_to_opencv_type(input_pixel_format_);
    if (opencv_input_type == -1) {
        running_ = false;
        return;
    }

    ImageData input_image;
    while (running_.load()) {
        if (input_queue_.wait_pop(input_image, std::chrono::milliseconds(10))) {
            auto process_start_time = std::chrono::high_resolution_clock::now();
            
            if (frame_counter_++ % skip_factor_ != 0) {
                if (input_image.buffer) input_image.buffer.reset();
                continue;
            }

            if (!input_image.buffer || input_image.buffer->data.empty() || input_image.width == 0 || input_image.height == 0) {
                continue;
            }

            cv::Mat input_frame_mat;
            if (input_image.format == libcamera::formats::RGB888) {
                input_frame_mat = cv::Mat(input_image.height, input_image.width, CV_8UC3, input_image.buffer->data.data());
            } else if (input_image.format == libcamera::formats::YUYV) {
                cv::Mat yuyv_mat = cv::Mat(input_image.height, input_image.width, CV_8UC2, input_image.buffer->data.data());
                cv::cvtColor(yuyv_mat, input_frame_mat, cv::COLOR_YUV2BGR_YUYV, 3);
            } else {
                input_image.buffer.reset();
                continue;
            }

            cv::Mat processed_mat;
            if (input_image.width == (unsigned int)output_width_ && input_image.height == (unsigned int)output_height_) {
                processed_mat = input_frame_mat;
            } else {
                cv::resize(input_frame_mat, processed_mat, cv::Size(output_width_, output_height_), 0, 0, cv::INTER_NEAREST);
            }

            if (detection_buffer_ptr_ != nullptr) {
                if (detection_buffer_ptr_->update_consumer()) {
                    if (app_ref_) {
                        app_ref_->increment_inference_results_consumed_by_overlay();
                    }
                }
                const auto& detections = detection_buffer_ptr_->get_read_buffer();
                if (!detections.empty()) {
                    apply_detections_to_frame(processed_mat, detections);
                }
            }

            std::shared_ptr<PooledBuffer<uint8_t>> processed_buffer_data = buffer_pool_->acquire();
            if (processed_buffer_data) {
                size_t required_size = processed_mat.total() * processed_mat.elemSize();
                if (processed_buffer_data->data.size() < required_size) {
                    processed_buffer_data->data.resize(required_size);
                }
                
                if (!processed_mat.empty() && processed_mat.data != nullptr) {
                    std::memcpy(processed_buffer_data->data.data(), processed_mat.data, required_size);
                    processed_buffer_data->size = required_size;

                    ImageData output_image_data(input_image.timestamp_epoch_ms, input_image.frame_id);
                    output_image_data.width = output_width_;
                    output_image_data.height = output_height_;
                    output_image_data.format = libcamera::formats::RGB888;
                    output_image_data.buffer = processed_buffer_data;
                    output_image_data.fd = -1;
                    
                    output_queue_.push(std::move(output_image_data));
                }
            }
            
            if (input_image.buffer) input_image.buffer.reset();

            auto process_end_time = std::chrono::high_resolution_clock::now();
            long long preprocess_duration_us = std::chrono::duration_cast<std::chrono::microseconds>(process_end_time - process_start_time).count();
            avg_preprocess_time_us_.store(static_cast<long long>(avg_preprocess_time_us_.load() * 0.9 + preprocess_duration_us * 0.1));
        }
    }
}

void ImageProcessor::apply_detections_to_frame(cv::Mat& frame, const DetectionResults& detections) {
    int frame_width = frame.cols;
    int frame_height = frame.rows;

    for (const auto& detection : detections) {
        int x_min = std::max(0, static_cast<int>(detection.xmin * frame_width));
        int y_min = std::max(0, static_cast<int>(detection.ymin * frame_height));
        int x_max = std::min(frame_width - 1, static_cast<int>(detection.xmax * frame_width));
        int y_max = std::min(frame_height - 1, static_cast<int>(detection.ymax * frame_height));

        cv::Scalar box_color(0, 0, 255);
        cv::rectangle(frame, cv::Point(x_min, y_min), cv::Point(x_max, y_max), box_color, 2);
    }
}
