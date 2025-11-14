#!/usr/bin/env python3
# --- FORCE UNBUFFERED OUTPUT ---
import sys
import signal # Moved to top level
import time # Moved to top level
print("=== TPU Server Starting (Pre-import Phase) ===", flush=True)

# tpu_server.py
# Listens for camera frames on a Unix stream socket, runs inference on a Coral TPU,
# and sends detection results over UDP.

# --- Global Running Flag & Signal Handling for Graceful Shutdown ---
running = True
interpreter = None  # Global interpreter reference for cleanup

def cleanup_tpu():
    """Force release of TPU resources"""
    global interpreter
    if interpreter is not None:
        log("Releasing TPU resources...")
        interpreter = None
        time.sleep(0.1)  # Allow driver cleanup

def signal_handler(sig, frame):
    global running
    log("Signal received, shutting down...")
    running = False
    cleanup_tpu()  # Critical: release TPU on exit

signal.signal(signal.SIGINT, signal_handler)
signal.signal(signal.SIGTERM, signal_handler)


# --- Logging ---
def log(message, file=sys.stdout):
    """Prints a message with a UTC timestamp and flushes immediately."""
    timestamp = time.strftime('%Y-%m-%d %H:%M:%S', time.gmtime())
    print(f"[{timestamp}] {message}", file=file, flush=True)

log("Starting imports...")
try:
    import os
    log("  - os imported")
    import socket
    log("  - socket imported")
    import numpy as np
    log("  - numpy imported")
    import json
    log("  - json imported")
    # time is already imported above
    # signal is already imported above
    from PIL import Image
    log("  - PIL.Image imported successfully")
    from struct import unpack # For unpacking frame_id and frame_size
    log("  - struct.unpack imported")
    import pycoral.utils.edgetpu as edgetpu # Import module directly
    log("  - pycoral.utils.edgetpu imported")
    # Suppress the enormous amount of tflite logging
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    # The tflite_runtime package is installed via apt
    import tflite_runtime.interpreter as tflite
    log("  - tflite_runtime.interpreter imported")
    from pycoral.adapters import common
    log("  - pycoral.adapters.common imported")
    from pycoral.adapters import detect
    log("  - pycoral.adapters.detect imported")
except Exception as e:
    log(f"CRITICAL IMPORT ERROR: {e}", file=sys.stderr)
    sys.exit(1)

log("All imports successful.")

# --- Configuration ---
# Socket path (must match the volume mount in Docker)
UNIX_SOCK_PATH = "/app/socket/camera.sock"

# Frame dimensions (must match the C++ producer)
RAW_FRAME_WIDTH = 1536
RAW_FRAME_HEIGHT = 864
RAW_FRAME_BYTES = RAW_FRAME_WIDTH * RAW_FRAME_HEIGHT * 3 # RGB888

# Model and label files
MODEL_FILE = "/app/ssd_mobilenet_v2_coco_quant_postprocess_edgetpu.tflite"
LABEL_FILE = "/app/coco_labels.txt"

# Inference settings
SCORE_THRESHOLD = 0.5

def load_labels(path):
    """Loads labels from a file."""
    with open(path, 'r') as f:
        return {i: line.strip() for i, line in enumerate(f.readlines())}

def recv_exact(sock, size):
    data = bytearray()
    while len(data) < size:
        try:
            packet = sock.recv(size - len(data))
            if not packet:
                raise ConnectionError("Socket closed by peer during recv_exact")
            data.extend(packet)
        except socket.timeout:
            raise socket.timeout("Socket timeout during recv_exact")
        except BlockingIOError:
            # This should not happen with blocking sockets, but good to catch
            time.sleep(0.001) # Small sleep to prevent busy-waiting
            continue
        except Exception as e:
            raise Exception(f"Error during recv_exact: {e}")
    return bytes(data)

def main():
    """Main function to set up sockets, load model, and run inference loop."""
    log("--- Coral TPU Inference Server (Debian 10, SOCK_STREAM) ---")

    # --- Get Network Configuration from Environment ---
    phone_ip = os.getenv("PHONE_IP")
    # Use 9090 for UDP as per Android app's UdpBoxReceiver default
    phone_port = int(os.getenv("PHONE_PORT", 9090)) 

    if not phone_ip:
        log("Error: PHONE_IP environment variable not set.", file=sys.stderr)
        sys.exit(1)

    log(f"Streaming detection results to {phone_ip}:{phone_port}")

    # --- Load Model and Labels ---
    log("Checking for Edge TPU device...")
    if not os.path.exists("/dev/apex_0"):
        log("CRITICAL ERROR: /dev/apex_0 not found! Is the TPU connected and drivers loaded?", file=sys.stderr)
        sys.exit(1)
    log("Edge TPU device found at /dev/apex_0")

    # Add permissions check
    if not os.access("/dev/apex_0", os.R_OK | os.W_OK):
        log("CRITICAL ERROR: No read/write access to /dev/apex_0! Check device permissions.", file=sys.stderr)
        sys.exit(1)
    log("Edge TPU device is accessible")

    # Now load model (with timeout protection)
    log("Loading model and labels...")
    global interpreter  # Declare as global
    try:
        labels = load_labels(LABEL_FILE)
        
        # Add a timeout alarm for the TPU init (Unix-only)
        def timeout_handler(signum, frame):
            raise TimeoutError("TPU initialization timed out")
        
        # Set 30-second timeout (was 10)
        old_handler = signal.signal(signal.SIGALRM, timeout_handler)
        signal.alarm(30)
        
        try:
            interpreter = edgetpu.make_interpreter(MODEL_FILE)
            interpreter.allocate_tensors()
            log("Model loaded successfully.")
        finally:
            signal.alarm(0)  # Disable alarm
            signal.signal(signal.SIGALRM, old_handler)
            
    except TimeoutError as e:
        log(f"ERROR: {e}", file=sys.stderr)
        log("The Edge TPU is not responding. Check dmesg for driver errors.", file=sys.stderr)
        sys.exit(1)
    except Exception as e:
        log(f"Error loading model: {e}", file=sys.stderr)
        log("This might be because the Edge TPU device is not found or the driver is not installed.", file=sys.stderr)
        sys.exit(1)


    input_height, input_width = common.input_size(interpreter)

    # --- Set up Sockets ---
    # Unix stream socket to receive frames from the host
    unix_sock_server = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
    try:
        unix_sock_server.setsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF, 32 * 1024 * 1024)
        actual_rcvbuf = unix_sock_server.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
        log(f"RCVBUF requested: {32 * 1024 * 1024} bytes, actual: {actual_rcvbuf} bytes")
    except Exception as e:
        log(f"Failed to set RCVBUF: {e}", file=sys.stderr)
    # UDP socket to send results to the phone
    udp_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)

    # Clean up old socket file if it exists
    if os.path.exists(UNIX_SOCK_PATH):
        os.remove(UNIX_SOCK_PATH)

    try:
        unix_sock_server.bind(UNIX_SOCK_PATH)
        unix_sock_server.listen(1)
        log(f"Listening for a connection on {UNIX_SOCK_PATH}")

        while running:
            log("Waiting for a new client connection...")
            try:
                conn, addr = unix_sock_server.accept()
                conn.settimeout(5.0) # Set timeout for client connection
                log("Client connected.")
                # Verify actual RCVBUF after connection
                actual_rcvbuf_conn = conn.getsockopt(socket.SOL_SOCKET, socket.SO_RCVBUF)
                log(f"ACTUAL RCVBUF on connected socket: {actual_rcvbuf_conn} bytes")
            except socket.error as e:
                # Handle potential errors during accept, e.g., if socket is closed during shutdown
                if e.errno == socket.errno.ECONNABORTED or e.errno == socket.errno.ECONNRESET:
                    log("Connection aborted or reset by peer.")
                    continue # Try accepting again
                elif e.errno == socket.errno.EAGAIN or e.errno == socket.errno.EWOULDBLOCK: # EWOULDBLOCK could be here
                    # This is for non-blocking sockets, but good to have if context changes
                    time.sleep(0.01) # Small sleep to prevent busy-waiting
                    continue
                else:
                    log(f"Error accepting connection: {e}", file=sys.stderr)
                    break # Exit loop on critical error

            try:
                # --- Inference Loop ---
                while running:
                    # Receive frame_id (4 bytes, uint32_t, big-endian)
                    frame_id_data = recv_exact(conn, 4)
                    if frame_id_data is None:
                        if not running: break
                        log("Client disconnected (no frame_id).")
                        break
                    frame_id = unpack('>I', frame_id_data)[0] # '>I' for big-endian unsigned int

                    # Receive frame size (4 bytes, uint32_t, big-endian)
                    frame_size_data = recv_exact(conn, 4)
                    if frame_size_data is None:
                        if not running: break
                        log("Client disconnected (no frame_size).")
                        break
                    frame_size = unpack('>I', frame_size_data)[0]

                    # Validate frame size to ensure producer and consumer are in sync
                    if frame_size != RAW_FRAME_BYTES:
                        log(f"Error: Invalid frame size received. Expected {RAW_FRAME_BYTES}, got {frame_size}. Closing connection.", file=sys.stderr)
                        break
                    
                    # Receive the actual frame data
                    data = recv_exact(conn, frame_size)
                    if data is None:
                        if not running: break
                        log("Client disconnected (no frame data).")
                        break

                    log(f"  - Received frame {frame_id} with size {len(data)}")

                    if len(data) != frame_size:
                        log(f"Warning: Received incomplete frame. Got {len(data)}, expected {frame_size}", file=sys.stderr)
                        continue

                    # --- Pre-processing ---
                    # The data is coming in as a flat RGB888 array.
                    # Convert to a Pillow Image for high-quality resizing.
                    image = Image.frombytes('RGB', (RAW_FRAME_WIDTH, RAW_FRAME_HEIGHT), bytes(data))
                    
                    # Resize the image to the model's input dimensions
                    resized_image = image.resize((input_width, input_height))
                    
                    # Convert back to a NumPy array for the interpreter
                    rgb_image = np.array(resized_image)

                    # --- Run Inference ---
                    common.set_input(interpreter, rgb_image)
                    interpreter.invoke()
                    objs = detect.get_objects(interpreter, SCORE_THRESHOLD, (1.0, 1.0))

                    # --- Format and Send Results ---
                    results = []
                    for obj in objs:
                        bbox = obj.bbox
                        results.append({
                            "class": int(obj.id),
                            "label": labels.get(obj.id, "unknown"),
                            "score": float(obj.score),
                            "box": [bbox.ymin, bbox.xmin, bbox.ymax, bbox.xmax]
                        })

                    if results:
                        log(f"Frame {frame_id}: Detected {len(results)} objects")
                        for res in results:
                            log(f"  - {res['label']}: {res['score']:.2f} @ {res['box']}")

                    # Include the frame_id in the JSON payload
                    payload = {
                        "frame_id": int(frame_id), # Ensure frame_id is an int for JSON
                        "detections": results
                    }
                    json_payload = json.dumps(payload).encode('utf-8')
                    udp_sock.sendto(json_payload, (phone_ip, phone_port))

                    # Flush any backed up data (non-blocking read)
                    conn.setblocking(False)
                    try:
                        while True:
                            # Read up to 4KB at a time to clear the buffer
                            chunk = conn.recv(4096)
                            if not chunk:
                                break # No more data to read
                    except BlockingIOError:
                        pass # Expected when no more data
                    finally:
                        conn.setblocking(True) # Restore blocking mode
            
            except socket.timeout:
                log("Socket timeout on client connection. Checking for shutdown.", file=sys.stderr)
                if not running:
                    break
            except ConnectionError as e:
                log(f"Connection error: {e}", file=sys.stderr)
            except Exception as e:
                log(f"An error occurred during inference: {e}", file=sys.stderr)
            finally:
                conn.close()
                log("Client connection closed.")

    except KeyboardInterrupt:
        log("\nCaught KeyboardInterrupt, shutting down.")
    except Exception as e:
        log(f"A server error occurred: {e}", file=sys.stderr)
    finally:
        log("Cleaning up sockets...")
        unix_sock_server.close()
        udp_sock.close()
        if os.path.exists(UNIX_SOCK_PATH):
            os.remove(UNIX_SOCK_PATH)
        cleanup_tpu()  # Ensure TPU is released
        log("Server stopped.")

if __name__ == '__main__':
    main()
