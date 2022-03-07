#!/usr/bin/env python3

import socket
import logging
import struct
import select
from controller import VX360CanGamepad

IP = "0.0.0.0"
PORT = 5454

def handle_traffic(client_socket):
    """
    Create a generator that receives messages over udp and converts them to
    joystick commands. The udp messages consist of an id (2 bytes, as CAN-ids are 11 bits),
    in combination with at most 8 bytes of data.
    """
    # Create a Struct object for faster unpacking
    unsignedShortStruct = struct.Struct('>H')
    while True:
        # Use a timeout to ensure the program stays (semi)responsive.
        readable, _, _ = select.select([client_socket], [], [], 0.5)
        for readable_socket in readable:
            data = readable_socket.recv(10)
            # Combine the first two bytes to get the arbitration id
            arbitration_id, = unsignedShortStruct.unpack(data[:2])
            yield arbitration_id, data[2:]
    
def main():
    logging.info('Creating virtual controller device...')
    controller = VX360CanGamepad()
    logging.info('Controller created.')
    logging.info(f'Binding on port {PORT} on ip address {IP}')
    with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as client_socket:
        client_socket.bind((IP, PORT))
        logging.info('Ready!')
        for arbitration_id, data in handle_traffic(client_socket):
            controller.handle(arbitration_id, data)


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.StreamHandler()
        ]
    )
    main()
