#include <gtest/gtest.h>
#include "../src/config_loader.h"
#include <fstream>
#include <string>
#include <filesystem>

// Define a temporary config file path for testing
const std::string TEST_CONFIG_PATH = "test_config.json";

// Helper function to create a temporary config file
void create_test_config_file(const std::string& content) {
    std::ofstream file(TEST_CONFIG_PATH);
    ASSERT_TRUE(file.is_open()) << "Failed to create temporary config file.";
    file << content;
    file.close();
}

// Helper function to clean up the temporary config file
void cleanup_test_config_file() {
    std::filesystem::remove(TEST_CONFIG_PATH);
}

// Test fixture for ConfigLoader tests
class ConfigLoaderTest : public ::testing::Test {
protected:
    void SetUp() override {
        // Ensure no previous test file exists
        cleanup_test_config_file();
    }

    void TearDown() override {
        // Clean up after each test
        cleanup_test_config_file();
    }
};

// Test case for successful loading of a valid config file
TEST_F(ConfigLoaderTest, LoadValidConfigFile) {
    std::string config_content = R"({
        "application": {
            "model_path": "test_model.tflite",
            "labels_path": "test_labels.txt",
            "listen_address": "127.0.0.1",
            "high_res_width": 1024,
            "high_res_height": 768,
            "camera_watchdog_timeout_seconds": 5,
            "inference_worker_threads": 4,
            "jpeg_quality": 85,
            "camera_fps": 60.0,
            "detection_score_threshold": 0.7,
            "log_path": "/tmp/test_logs",
            "video_stream": {
                "protocol": "RTSP",
                "address": "192.168.0.100",
                "port": 554
            },
            "telemetry": {
                "protocol": "ZeroMQ",
                "pub_address": "tcp://*:7000"
            }
        }
    })";
    create_test_config_file(config_content);

    ConfigLoader loader;
    ASSERT_TRUE(loader.load(TEST_CONFIG_PATH));

    EXPECT_EQ(loader.get_model_path(), "test_model.tflite");
    EXPECT_EQ(loader.get_labels_path(), "test_labels.txt");
    EXPECT_EQ(loader.get_listen_address(), "127.0.0.1");
    EXPECT_EQ(loader.get_high_res_width(), 1024);
    EXPECT_EQ(loader.get_high_res_height(), 768);
    EXPECT_EQ(loader.get_camera_watchdog_timeout().count(), 5);
    EXPECT_EQ(loader.get_inference_worker_threads(), 4);
    EXPECT_EQ(loader.get_jpeg_quality(), 85);
    EXPECT_EQ(loader.get_camera_fps(), 60.0);
    EXPECT_EQ(loader.get_detection_score_threshold(), 0.7f);
    EXPECT_EQ(loader.get_log_path(), "/tmp/test_logs");

    // New video stream getters
    EXPECT_EQ(loader.get_video_stream_protocol(), "RTSP");
    EXPECT_EQ(loader.get_video_stream_address(), "192.168.0.100");
    EXPECT_EQ(loader.get_video_stream_port(), 554);

    // New telemetry getters
    EXPECT_EQ(loader.get_telemetry_protocol(), "ZeroMQ");
    EXPECT_EQ(loader.get_telemetry_pub_address(), "tcp://*:7000");


}

// Test case for missing config file
TEST_F(ConfigLoaderTest, LoadMissingConfigFile) {
    ConfigLoader loader;
    ASSERT_FALSE(loader.load("non_existent_config.json"));
}

// Test case for invalid JSON content
TEST_F(ConfigLoaderTest, LoadInvalidJsonFile) {
    std::string config_content = "{ invalid json }";
    create_test_config_file(config_content);

    ConfigLoader loader;
    ASSERT_FALSE(loader.load(TEST_CONFIG_PATH));
}

// Test case for default values when keys are missing
TEST_F(ConfigLoaderTest, DefaultValuesForMissingKeys) {
    std::string config_content = R"({
        "application": {}
    })";
    create_test_config_file(config_content);

    ConfigLoader loader;
    ASSERT_TRUE(loader.load(TEST_CONFIG_PATH));

    // Check some default values (from src/config_loader.cpp)
    EXPECT_EQ(loader.get_model_path(), "model.tflite");
    EXPECT_EQ(loader.get_high_res_width(), 1920);
    EXPECT_EQ(loader.get_video_stream_protocol(), "HTTP_WEBSOCKET"); // Default for new getter
    EXPECT_EQ(loader.get_telemetry_pub_address(), "tcp://*:6000"); // Default for new getter
}
