import zmq
import json
import time
import socket
import threading

BEACON_PORT = 5678
ORIENTATION_PORT = 5555
VIDEO_PORT = 5000

def beacon_sender():
    print(f"[*] Starting beacon sender on port {BEACON_PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, SO_BROADCAST, 1)
    
    beacon = {
        "type": "ANDROID_CORAL_CONTROLLER",
        "name": "MockAndroidApp",
        "orientation_port": ORIENTATION_PORT,
        "video_port": VIDEO_PORT
    }
    
    msg = json.dumps(beacon).encode('utf-8')
    while True:
        sock.sendto(msg, ('255.255.255.255', BEACON_PORT))
        time.sleep(2)

def orientation_sender():
    print(f"[*] Starting orientation sender on port {ORIENTATION_PORT}...")
    context = zmq.Context()
    socket = context.socket(zmq.PUB)
    socket.bind(f"tcp://*:{ORIENTATION_PORT}")
    
    yaw, pitch, roll = 0.0, 0.0, 0.0
    while True:
        data = {
            "yaw": yaw,
            "pitch": pitch,
            "roll": roll
        }
        socket.send_string(json.dumps(data))
        
        yaw = (yaw + 1.0) % 360
        pitch = (pitch + 0.5) % 90
        roll = (roll + 0.2) % 180
        
        time.sleep(0.1) # 10Hz

def video_receiver():
    print(f"[*] Starting video receiver on port {VIDEO_PORT}...")
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.bind(('0.0.0.0', VIDEO_PORT))
    
    count = 0
    start_time = time.time()
    while True:
        data, addr = sock.recvfrom(65535)
        count += len(data)
        if time.time() - start_time > 5:
            print(f"[+] Received {count / (1024*1024):.2f} MB of video data from {addr}")
            start_time = time.time()
            count = 0

if __name__ == "__main__":
    # Check for SO_BROADCAST constant if not defined in socket module
    if not hasattr(socket, 'SO_BROADCAST'):
        socket.SO_BROADCAST = 6 # Common value on Linux
        
    t1 = threading.Thread(target=beacon_sender, daemon=True)
    t2 = threading.Thread(target=orientation_sender, daemon=True)
    t3 = threading.Thread(target=video_receiver, daemon=True)
    
    t1.start()
    t2.start()
    t3.start()
    
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        print("Stopping...")
