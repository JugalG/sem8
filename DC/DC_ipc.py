# ********************************** SERVER *********************************
# SERVER

import socket
import threading

def handle_client(client_socket):
    try:
        data = client_socket.recv(1024)
        if not data:
            return
        received_data = list(map(int, data.decode('utf-8').split(',')))
        print(f"Received integers: {received_data}")
        print(f"Sum of integers: {sum(received_data)}")
    except Exception as e:
        print(f"Error handling client data: {e}")
    finally:
        client_socket.close()

def start_server():
    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.bind(('0.0.0.0', 8888))
    server.listen(5)
    print("[*] Listening on 0.0.0.0:8888")
    try:
        while True:
            client_socket, addr = server.accept()
            print(f"[*] Accepted connection from {addr[0]}:{addr[1]}")
            client_handler = threading.Thread(target=handle_client, args=(client_socket,))
            client_handler.start()
    except Exception as e:
        print(f"Server error: {e}")
    finally:
        server.close()

if __name__ == "__main__":
    start_server()



# *************************************** CLIENT ************************************************
# CLIENT

import socket

def send_data(target_ip, target_port, data):
    try:
        client = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        client.connect((target_ip, target_port))
        data_str = ','.join(map(str, data))
        client.send(data_str.encode('utf-8'))
    except Exception as e:
        print(f"Error sending data: {e}")
    finally:
        client.close()

if __name__ == "__main__":
    try:
        print("Enter 4 integers to send to the server:")
        integers_to_send = []
        for i in range(1, 5):
            num = int(input(f"{i}: "))
            integers_to_send.append(num)
        send_data('127.0.0.1', 8888, integers_to_send)
    except Exception as e:
        print(f"Client error: {e}")
