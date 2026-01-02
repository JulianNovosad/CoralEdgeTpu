CXX = g++
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra -pthread
INCLUDES = -Isrc

# Dependencies using pkg-config
PKG_CONFIG_DEPS = opencv4 gstreamer-1.0 gstreamer-app-1.0 gstreamer-video-1.0 libcamera libzmq
CXXFLAGS += $(shell pkg-config --cflags $(PKG_CONFIG_DEPS))
LIBS += $(shell pkg-config --libs $(PKG_CONFIG_DEPS)) -latomic -lpthread -ltensorflow-lite -ledgetpu

# Source files
SRC_DIR = src
SOURCES = $(wildcard $(SRC_DIR)/*.cpp)
OBJECTS = $(SOURCES:.cpp=.o)
TARGET = detector

all: $(TARGET)

$(TARGET): $(OBJECTS)
	$(CXX) $(CXXFLAGS) -o $@ $^ $(LIBS)

%.o: %.cpp
	$(CXX) $(CXXFLAGS) $(INCLUDES) -c $< -o $@

clean:
	rm -f $(SRC_DIR)/*.o $(TARGET)

.PHONY: all clean
