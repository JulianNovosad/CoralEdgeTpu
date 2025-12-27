#!/usr/bin/env python3
"""
RTSP Stream Verification Script for CoralEdgeTpu Detector
This script programmatically tests the RTSP server to verify:
- SPS/PPS headers reception
- At least 5 consecutive IDR frames
- Stream stability over time
- Frame-level data integrity
"""

import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst, GLib
import time
import threading
import signal
import sys
import os

class RTSPStreamTester:
    def __init__(self):
        Gst.init(None)
        self.pipeline = None
        self.loop = None
        self.received_frames = []
        self.sps_received = False
        self.pps_received = False
        self.idr_frames_count = 0
        self.start_time = None
        self.test_duration = 15  # Test for 15 seconds
        self.test_completed = False
        self.error_occurred = False
        
    def create_pipeline(self, rtsp_url):
        """Create GStreamer pipeline to connect to RTSP stream"""
        pipeline_str = f"""
        rtspsrc location={rtsp_url} protocols=tcp ! 
        rtph264depay ! 
        h264parse ! 
        tee name=t
        t. ! queue ! avdec_h264 ! videoconvert ! appsink name=sink emit-signals=false sync=false
        t. ! queue ! fakesink sync=false
        """
        
        self.pipeline = Gst.parse_launch(pipeline_str)
        
        # Get the appsink element to access frames
        appsink = self.pipeline.get_by_name('sink')
        appsink.set_property('emit-signals', False)
        appsink.set_property('sync', False)
        
        # Connect to bus to catch messages
        bus = self.pipeline.get_bus()
        bus.add_signal_watch()
        bus.connect('message', self.on_message)
        
    def on_message(self, bus, message):
        """Handle messages from GStreamer"""
        if message.type == Gst.MessageType.EOS:
            print("End of stream received")
            self.test_completed = True
            if self.loop:
                self.loop.quit()
        elif message.type == Gst.MessageType.ERROR:
            err, debug = message.parse_error()
            print(f"Error: {err}, Debug: {debug}")
            self.error_occurred = True
            if self.loop:
                self.loop.quit()
        elif message.type == Gst.MessageType.STATE_CHANGED:
            if isinstance(message.src, Gst.Pipeline):
                old_state, new_state, pending_state = message.parse_state_changed()
                if new_state == Gst.State.PLAYING:
                    print("Pipeline is now PLAYING")
    
    def analyze_buffer(self, pad, info):
        """Analyze incoming buffer for H.264 frame types"""
        buf = info.get_buffer()
        if buf:
            success, map_info = buf.map(Gst.MapFlags.READ)
            if success:
                data = map_info.data
                if len(data) > 0:
                    # Check for H.264 NAL unit start codes (0x00000001 or 0x000001)
                    if len(data) >= 4 and data[0:4] == b'\x00\x00\x00\x01':
                        nal_unit_type = data[4] & 0x1F if len(data) > 4 else 0
                        self.process_nal_unit(nal_unit_type, data)
                    elif len(data) >= 3 and data[0:3] == b'\x00\x00\x01':
                        nal_unit_type = data[3] & 0x1F if len(data) > 3 else 0
                        self.process_nal_unit(nal_unit_type, data)
                
                # Store frame info for analysis
                frame_info = {
                    'timestamp': buf.pts if buf.pts != Gst.CLOCK_TIME_NONE else 0,
                    'size': len(data),
                    'data': data[:min(10, len(data))]  # First 10 bytes for verification
                }
                self.received_frames.append(frame_info)
                
                buf.unmap(map_info)
        
        return Gst.PadProbeReturn.OK
    
    def process_nal_unit(self, nal_unit_type, data):
        """Process different types of NAL units"""
        if nal_unit_type == 7:  # SPS
            print(f"SPS header received (size: {len(data)})")
            self.sps_received = True
        elif nal_unit_type == 8:  # PPS
            print(f"PPS header received (size: {len(data)})")
            self.pps_received = True
        elif nal_unit_type == 5:  # IDR Slice
            print(f"IDR frame received (size: {len(data)})")
            self.idr_frames_count += 1
        elif nal_unit_type == 6:  # SEI
            print(f"SEI frame received (size: {len(data)})")
    
    def start_test(self, rtsp_url):
        """Start the RTSP stream test"""
        print(f"Starting RTSP stream test for {self.test_duration} seconds...")
        print(f"Connecting to: {rtsp_url}")
        
        self.create_pipeline(rtsp_url)
        
        # Add probe to h264parse pad to analyze NAL units
        h264parse = self.pipeline.get_by_name('h264parse')
        if h264parse:
            sink_pad = h264parse.get_static_pad('sink')
            if sink_pad:
                sink_pad.add_probe(Gst.PadProbeType.BUFFER, self.analyze_buffer)
        
        # Set pipeline to playing
        ret = self.pipeline.set_state(Gst.State.PLAYING)
        if ret == Gst.StateChangeReturn.FAILURE:
            print("Failed to set pipeline to PLAYING")
            return False
        
        # Start main loop in a separate thread
        self.start_time = time.time()
        self.loop = GLib.MainLoop()
        
        def run_loop():
            try:
                self.loop.run()
            except:
                pass
        
        loop_thread = threading.Thread(target=run_loop)
        loop_thread.daemon = True
        loop_thread.start()
        
        # Wait for test duration or error
        start_wait = time.time()
        while time.time() - start_wait < self.test_duration and not self.error_occurred and not self.test_completed:
            time.sleep(0.1)
        
        # Stop the pipeline
        if self.pipeline:
            self.pipeline.set_state(Gst.State.NULL)
        
        if self.loop and self.loop.is_running():
            self.loop.quit()
        
        return True
    
    def get_results(self):
        """Get test results"""
        return {
            'sps_received': self.sps_received,
            'pps_received': self.pps_received,
            'idr_frames_count': self.idr_frames_count,
            'total_frames': len(self.received_frames),
            'error_occurred': self.error_occurred,
            'test_completed': self.test_completed,
            'frames_with_valid_data': sum(1 for f in self.received_frames if f['data'] and any(b != 0 for b in f['data']))
        }

def main():
    """Main function to run the RTSP stream test"""
    print("RTSP Stream Verification Test Starting...")
    
    # Check if port 8554 is available (should be if detector is running)
    import socket
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            s.settimeout(2)  # 2 second timeout
            result = s.connect_ex(('localhost', 8554))
            if result != 0:
                print("ERROR: Port 8554 is not available. Is the detector running?")
                return False
            else:
                print("Port 8554 is available.")
    except Exception as e:
        print(f"Port check error: {e}")
        # Continue anyway since the detector may be starting up
    
    # Create tester and run test
    tester = RTSPStreamTester()
    rtsp_url = "rtsp://127.0.0.1:8554/live"
    
    success = tester.start_test(rtsp_url)
    if not success:
        print("Failed to start RTSP test")
        return False
    
    # Get and print results
    results = tester.get_results()
    
    print("\n" + "="*50)
    print("RTSP STREAM VERIFICATION RESULTS")
    print("="*50)
    print(f"SPS header received: {results['sps_received']}")
    print(f"PPS header received: {results['pps_received']}")
    print(f"IDR frames received: {results['idr_frames_count']}/5 minimum")
    print(f"Total frames received: {results['total_frames']}")
    print(f"Frames with valid data: {results['frames_with_valid_data']}")
    print(f"Test completed successfully: {results['test_completed']}")
    print(f"Error occurred: {results['error_occurred']}")
    
    # Check if requirements are met
    requirements_met = (
        results['sps_received'] and
        results['pps_received'] and
        results['idr_frames_count'] >= 5 and
        results['frames_with_valid_data'] > 0 and
        not results['error_occurred']
    )
    
    print("\nREQUIREMENTS CHECK:")
    print(f"- SPS/PPS headers: {'PASS' if results['sps_received'] and results['pps_received'] else 'FAIL'}")
    print(f"- Minimum 5 IDR frames: {'PASS' if results['idr_frames_count'] >= 5 else 'FAIL'}")
    print(f"- Valid frame data: {'PASS' if results['frames_with_valid_data'] > 0 else 'FAIL'}")
    print(f"- No errors: {'PASS' if not results['error_occurred'] else 'FAIL'}")
    
    print("\nOVERALL RESULT:", "PASS" if requirements_met else "FAIL")
    print("="*50)
    
    return requirements_met

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)