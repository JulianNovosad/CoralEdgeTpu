import asyncio
import websockets
import sys

async def receive_h264_stream(uri, output_file):
    """
    Connects to the WebSocket URI and saves incoming binary packets to a file.
    """
    print(f"Attempting to connect to {uri}...")
    try:
        async with websockets.connect(uri, ping_interval=None, ping_timeout=None) as websocket:
            print(f"Successfully connected to WebSocket stream: {uri}")
            print(f"Saving incoming H.264 packets to '{output_file}'...")
            packet_count = 0
            try:
                while True:
                    packet = await websocket.recv()
                    if isinstance(packet, bytes):
                        with open(output_file, "ab") as f:
                            f.write(packet)
                        packet_count += 1
                        print(f"Received packet {packet_count} | Size: {len(packet)} bytes")
                    else:
                        print(f"Received non-binary message: {packet}")

            except websockets.ConnectionClosed as e:
                print(f"Connection closed by server: {e.reason} (Code: {e.code})")
            except Exception as e:
                print(f"An error occurred while receiving data: {e}")

    except ConnectionRefusedError:
        print(f"Connection refused. Is the detector running on the host and listening on the correct port?")
    except Exception as e:
        print(f"Failed to connect to WebSocket server: {e}")

if __name__ == "__main__":
    # --- Configuratie ---
    # Vervang dit IP-adres door het adres van je Raspberry Pi
    HOST_IP = "192.168.178.48"
    PORT = 8080
    
    if len(sys.argv) > 1:
        HOST_IP = sys.argv[1]
        print(f"Using provided IP address: {HOST_IP}")

    WEBSOCKET_URI = f"ws://{HOST_IP}:{PORT}/stream"
    OUTPUT_FILE = "received_stream.h264"
    
    # Maak het uitvoerbestand leeg bij start
    with open(OUTPUT_FILE, "wb") as f:
        pass

    print("-" * 50)
    print(f"WebSocket H.264 Client")
    print(f"Connecting to: {WEBSOCKET_URI}")
    print(f"Output file:   {OUTPUT_FILE}")
    print("-" * 50)

    try:
        asyncio.run(receive_h264_stream(WEBSOCKET_URI, OUTPUT_FILE))
    except KeyboardInterrupt:
        print("\nScript stopped by user.")
