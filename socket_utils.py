import socket
import pickle
import struct  # Used to send fixed-size headers

def send_data(data, addr):
    """Send serialized data over a socket with a length prefix."""
    serialized_data = pickle.dumps(data)  
    data_length = struct.pack("!I", len(serialized_data))  # 4-byte length header

    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.connect(addr)
        s.sendall(data_length + serialized_data)  # Send length + data

def receive_data(port, timeout=None):
    """Receive serialized data over a socket, ensuring complete transfer."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)  # Allow reusing the port
        s.bind(("0.0.0.0", port))
        s.listen(1)

        if timeout:
            s.settimeout(timeout)

        try:
            conn, _ = s.accept()
            with conn:
                data_length = struct.unpack("!I", conn.recv(4))[0]  # Read the 4-byte length header
                received_data = b""

                while len(received_data) < data_length:  # Keep receiving until full data is read
                    packet = conn.recv(4096)  # Read in chunks
                    if not packet:
                        break
                    received_data += packet

                return pickle.loads(received_data) if received_data else None

        except socket.timeout:
            return None
