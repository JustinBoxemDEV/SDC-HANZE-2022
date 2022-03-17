#/usr/bin/env python3

import sys
import cv2
import can
import os
import threading
import time
import struct
from datetime import datetime
from typing import Optional
from queue import Queue

CAN_MSG_SENDING_SPEED = .040 # 25Hz

class CanListener:
    """
    A can listener that listens for specific messages and stores their latest values.
    """

    _id_conversion = {
        0x12c: 'steering',
        0x120: 'throttle',
        0x126: 'brake'
    }

    def __init__(self, bus: can.Bus):
        self.bus = bus
        self.thread = threading.Thread(target = self._listen, args = (), daemon = True)
        self.running = False
        self.data = {'steering': None, 'throttle': None, 'brake': None}
    
    def start_listening(self):
        self.running = True
        self.thread.start()
    
    def stop_listening(self):
        self.running = False
    
    def get_new_values(self):
        values = self.data
        return values

    def _listen(self):
        while self.running:
            message: Optional[can.Message] = self.bus.recv(.5)
            if message and message.arbitration_id in self._id_conversion:
                self.data[self._id_conversion[message.arbitration_id]] = message.data

class ImageWorker:
    """
    A worker that writes images to disk.
    """

    def __init__(self, image_queue: Queue, folder_name):
        self.queue = image_queue
        self.thread = threading.Thread(target = self._process, args = (), daemon = True)
        self.folder_name = folder_name
    
    def start(self):
        self.thread.start()
    
    def stop(self):
        self.queue.join()

    def put(self, data):
        self.queue.put(data)
    
    def _process(self):
        while True:
            timestamp, image = self.queue.get()
            cv2.imwrite(self.folder_name + f'/{timestamp}.jpg', image)
            self.queue.task_done()

class CanWorker:
    """
    A worker that writes can-message values to disk.
    """

    def __init__(self, can_queue: Queue, folder_name):
        self.queue = can_queue
        self.thread = threading.Thread(target = self._process, args = (), daemon = True)
        self.folder_name = folder_name
        self.file_pointer = open('data ' + self.folder_name + '.csv', 'w')
        print('Steering|Throttle|Brake|Speed|Image', file = self.file_pointer)
    
    def start(self):
        self.thread.start()
    
    def stop(self):
        self.queue.join()
        self.file_pointer.close()
    
    def put(self, data):
        self.queue.put(data)
    
    def _process(self):
        while True:
            timestamp, values = self.queue.get()
            steering = str(struct.unpack("f", bytearray(values["steering"][:4]))[0]) if values["steering"] else 0.0
            throttle = str(values["throttle"][0]/100) if values["throttle"] else 0.0
            brake = str(values["brake"][0]/100) if values["brake"] else 0.0
            print(f'{steering}|{throttle}|{brake}|\"' + self.folder_name + f'/{timestamp}.jpg\"', file=self.file_pointer)
            self.queue.task_done()


def main():
    folder_name = "images " + datetime.now().strftime("%d-%m-%Y %H-%M-%S")
    print('Initializing...', file=sys.stderr)
    if not os.path.exists(folder_name):
        os.mkdir(folder_name)
    bus = initialize_can()
    camera = initialize_camera()
    can_listener = CanListener(bus)
    can_listener.start_listening()
    image_queue = Queue()
    image_worker = ImageWorker(image_queue, folder_name)
    ImageWorker(image_queue, folder_name).start()
    ImageWorker(image_queue, folder_name).start()
    image_worker.start()
    can_worker = CanWorker(Queue(),folder_name)
    can_worker.start()

    print('Recording...', file=sys.stderr)
    try:
        while True:
            _, frame = camera.read()
            values = can_listener.get_new_values()
            timestamp = time.time()
            image_worker.put((timestamp, frame))
            can_worker.put((timestamp, values))
    except KeyboardInterrupt:
        pass
    
    print('Stopping...', file=sys.stderr)
    can_listener.stop_listening()
    image_worker.stop()
    can_worker.stop()


def initialize_can() -> can.Bus:
    """
    Set up the can bus interface and apply filters for the messages we're interested in.
    """
    bus = can.Bus(interface='socketcan', channel='can0', bitrate=500000)
    bus.set_filters([
        {'can_id': 0x12c, 'can_mask': 0xfff, 'extended': True}, # Steering
        {'can_id': 0x120, 'can_mask': 0xfff, 'extended': True}, # Throttle
        {'can_id': 0x126, 'can_mask': 0xfff, 'extended': True} # Brake
    ])
    return bus

def initialize_camera() -> cv2.VideoCapture:
    """
    Initialize the opencv camera capture device.
    """
    capture = cv2.VideoCapture(0)
    capture.set(cv2.CAP_PROP_FRAME_WIDTH, 848)
    capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    capture.set(cv2.CAP_PROP_AUTOFOCUS, 0)
    capture.set(cv2.CAP_PROP_FOCUS, 0)
    capture.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG')) #important to set right codec and enable the 60fps
    capture.set(cv2.CAP_PROP_FPS, 30) #make 60 to enable 60FPS
    return capture

if __name__ == '__main__':
    main()
