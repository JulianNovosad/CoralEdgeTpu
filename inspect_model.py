#!/usr/bin/env python3

import flatbuffers
import numpy as np

# Simple script to inspect the TFLite model structure
def inspect_model(model_path):
    try:
        with open(model_path, 'rb') as f:
            buf = bytearray(f.read())
        
        # Print basic info about the model file
        print(f"Model file size: {len(buf)} bytes")
        
        # Look for the magic number at the beginning
        if len(buf) >= 4:
            magic = int.from_bytes(buf[0:4], byteorder='little')
            print(f"Magic number: 0x{magic:08x}")
            
        # Look for version info
        if len(buf) >= 8:
            version = int.from_bytes(buf[4:8], byteorder='little')
            print(f"Version: {version}")
            
        print("Model inspection completed.")
        
    except Exception as e:
        print(f"Error inspecting model: {e}")

if __name__ == "__main__":
    inspect_model("detect_int8_edgetpu.tflite")