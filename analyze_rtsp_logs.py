#!/usr/bin/env python3
"""
RTSP Stream Verification Report for CoralEdgeTpu Detector
This script analyzes the detector logs to verify RTSP stream stability.
"""

import re
import sys

def analyze_rtsp_logs(log_file_path):
    """Analyze detector logs to verify RTSP stream stability"""
    with open(log_file_path, 'r', encoding='utf-8', errors='ignore') as f:
        log_content = f.read()
    
    # Extract RTSP-related information
    rtsp_start = re.search(r'RTSP server started successfully', log_content)
    rtsp_url = re.search(r'stream URL: (rtsp://[^\s]+)', log_content)
    
    # Count different frame types
    sps_frames = len(re.findall(r'NAL Type=SPS \(7\)', log_content))
    pps_frames = len(re.findall(r'NAL Type=PPS \(8\)', log_content))
    idr_frames = len(re.findall(r'NAL Type=IDR-Slice \(5\)', log_content))
    sei_frames = len(re.findall(r'NAL Type=SEI \(6\)', log_content))
    p_frames = len(re.findall(r'NAL Type=P-Slice \(1\)', log_content))
    
    # Check throughput information
    throughput_matches = re.findall(r'RTSP_THROUGHPUT: In=([\d.]+) fps', log_content)
    if throughput_matches:
        throughput_values = [float(x) for x in throughput_matches]
        avg_throughput = sum(throughput_values) / len(throughput_values) if throughput_values else 0
    else:
        avg_throughput = 0
    
    # Check for errors
    errors = re.findall(r'(ERROR|error|Error).*?RTSP', log_content, re.IGNORECASE)
    
    results = {
        'rtsp_server_started': bool(rtsp_start),
        'rtsp_url': rtsp_url.group(1) if rtsp_url else None,
        'sps_frames_count': sps_frames,
        'pps_frames_count': pps_frames,
        'idr_frames_count': idr_frames,
        'sei_frames_count': sei_frames,
        'p_frames_count': p_frames,
        'avg_throughput': avg_throughput,
        'errors_found': len(errors),
        'total_h264_frames': sps_frames + pps_frames + idr_frames + sei_frames + p_frames
    }
    
    return results

def print_verification_report(results):
    """Print a comprehensive verification report"""
    print("="*60)
    print("RTSP STREAM VERIFICATION REPORT")
    print("="*60)
    
    print(f"RTSP Server Started: {'YES' if results['rtsp_server_started'] else 'NO'}")
    print(f"RTSP URL: {results['rtsp_url']}")
    print()
    
    print("H.264 FRAME ANALYSIS:")
    print(f"  SPS headers: {results['sps_frames_count']}")
    print(f"  PPS headers: {results['pps_frames_count']}")
    print(f"  IDR frames: {results['idr_frames_count']}")
    print(f"  SEI frames: {results['sei_frames_count']}")
    print(f"  P frames: {results['p_frames_count']}")
    print(f"  Total H.264 frames: {results['total_h264_frames']}")
    print()
    
    print(f"Average Throughput: {results['avg_throughput']:.2f} fps")
    print(f"Errors found: {results['errors_found']}")
    print()
    
    # Requirements check
    requirements = [
        ("RTSP server started", results['rtsp_server_started']),
        ("SPS headers present", results['sps_frames_count'] > 0),
        ("PPS headers present", results['pps_frames_count'] > 0),
        ("At least 1 IDR frame (keyframe)", results['idr_frames_count'] >= 1),
        ("Valid H.264 frames generated", results['total_h264_frames'] > 5),
        ("No RTSP-related errors", results['errors_found'] == 0)
    ]
    
    print("REQUIREMENTS CHECK:")
    all_passed = True
    for req, passed in requirements:
        status = "PASS" if passed else "FAIL"
        print(f"  {req}: {status}")
        if not passed:
            all_passed = False
    
    print()
    print("OVERALL RESULT:", "RTSP STREAM VERIFIED STABLE" if all_passed else "RTSP STREAM VERIFICATION FAILED")
    print("="*60)
    
    return all_passed

def main():
    log_file_path = "detector_output.log"
    
    try:
        results = analyze_rtsp_logs(log_file_path)
        success = print_verification_report(results)
        return success
    except FileNotFoundError:
        print(f"Error: Log file {log_file_path} not found")
        return False
    except Exception as e:
        print(f"Error analyzing log file: {e}")
        return False

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)