import vgamepad
import struct
import time

class VX360CanGamepad(vgamepad.VX360Gamepad):
    """
    Create a virtual Xbox360 controller that is controlled by calling the
    handle method with CAN-like messages.
    """

    # Creating a struct object has performance benefits
    floatStruct = struct.Struct('f')

    def __init__(self):
        super().__init__()
    
    def handle(self, arbitration_id, data):
        """
        Convert data to virtual joystick movements based on the arbitration id (like can messages) and
        the data that would normally be sent over the can bus. There is no need for homing or extended
        ids. Homing messages will be ignored.
        """
        if arbitration_id == 0x120:
            self.throttle_float(VX360CanGamepad.convert_throttle_data(data))
        elif arbitration_id == 0x126:
            self.brake_float(VX360CanGamepad.convert_brake_data(data))
        elif arbitration_id == 0x12c:
            self.steer_float(VX360CanGamepad.convert_steering_data(data))
        elif arbitration_id == 0x121:
            self.gearshiftup()
        elif arbitration_id == 0x122:
            self.gearshiftdown()
        self.update()    

    @staticmethod
    def convert_throttle_data(data):
        """
        Convert a throttle message to a float between 0 (no throttle) and 1 (full throttle).
        Driving backwards is, unfortunately, NOT SUPPORTED yet.
        """
        # print(f"Data (first byte) when it reaches the controller: {data[0]}")
        throttle_percentage = data[0] # Throttle between 0 and 100 (incl)
        # print(f"Throttle percentage recieved by controller: {throttle_percentage/100}")
        return throttle_percentage/100

    @staticmethod
    def convert_brake_data(data):
        """
        Convert a brake message to a float between 0 (no braking) and 1 (fully applied brakes).
        """
        brake_percentage = data[0] # Brake between 0 and 100 (incl)
        return brake_percentage/100

    @staticmethod
    def convert_steering_data(data):
        """
        Convert a steering message to a float between -1 (left) and 1 (right).
        """
        # Steering is encoded as a float in the first four bytes of the data.
        steering_value, = VX360CanGamepad.floatStruct.unpack(bytearray(data[:4]))
        return steering_value

    def throttle_float(self, throttle_percentage):
        """
        Apply throttle. Throttle is mapped to the right trigger.
        """
        # 255 is the maximum value of a byte
        self.report.bRightTrigger = round(255 * throttle_percentage)
    
    def steer_float(self, steer_float):
        """
        Apply steering. Steering is mapped to the left thumbstick.
        """
        # 32767 is the maximum value of a signed short
        self.report.sThumbLX = round(32767 * steer_float)
    
    def brake_float(self, brake_percentage):
        """
        Apply braking. Braking is mapped to the left trigger.
        """
        # 255 is the maximum value of a byte
        self.report.bLeftTrigger = round(255 * brake_percentage)

    def gearshiftup(self):
        self.press_button(button=0x1000)
        self.update()
        time.sleep(0.5)
        self.release_button(button=0x1000)
        self.update()
        time.sleep(0.5)

    def gearshiftdown(self):
        self.press_button(button=0x4000)
        self.update()
        time.sleep(0.5)
        self.release_button(button=0x4000)
        self.update()
        time.sleep(0.5)