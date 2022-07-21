from inputs import get_gamepad
import inputs
import math
import sys
import threading
import can
from time import sleep
import struct
from os import system
from datetime import datetime

CAN_MSG_SENDING_SPEED = .04 # 100Hz

class XboxController(object):
	MAX_TRIG_VAL = math.pow(2, 8)
	MAX_JOY_VAL = math.pow(2, 15)


	def __init__(self):
		self.gamepad = inputs.devices.gamepads[0]

		self.LeftJoystickY = 0
		self.LeftJoystickX = 0
		self.RightJoystickY = 0
		self.RightJoystickX = 0
		self.LeftTrigger = 0
		self.RightTrigger = 0
		self.LeftBumper = 0
		self.RightBumper = 0
		self.A = 0
		self.X = 0
		self.Y = 0
		self.B = 0
		self.LeftThumb = 0
		self.RightThumb = 0
		self.Back = 0
		self.Start = 0
		self.LeftDPad = 0
		self.RightDPad = 0
		self.UpDPad = 0
		self.DownDPad = 0

		self._monitor_thread = threading.Thread(target=self._monitor_controller, args=())
		self._monitor_thread.daemon = True
		self._monitor_thread.start()
		
	def vibrate(self, duration=1000):
		try:
			self.gamepad.set_vibration(1,1,duration)
		except:
			print("Failed to vibrate")

	def read(self): # return the buttons/triggers that you care about in this methode
	
		def _rerange(x, in_min, in_max, out_min, out_max): 
			return float((x - in_min) * (out_max - out_min) / (in_max - in_min) + out_min)
			
		throttle = int(round(_rerange((self.RightTrigger * 100),0,400,0,57),0)) #reranges the values to 0-100
		brake = int(round(_rerange((self.LeftTrigger * 100),0,400,0,100),0)) #reranges the values to 0-100

		if(self.LeftJoystickX * 100 > 10):
			steering = round(_rerange((self.LeftJoystickX * 10),1,10,0,1),5)
		elif (self.LeftJoystickX * 100 < -10):
			steering = round(_rerange((self.LeftJoystickX * 10),-1,-10,-0,-1),5)
		else:
			steering = 0
		#steering = min(max(steering ,-1),1)/2 # Set steer between -1 and 1, then scale
		a = self.A
		y = self.Y
		x = self.X
		b = self.B
		#b = self.X # b=1, x=2
		rb = self.RightBumper
		
		return [throttle, brake, steering/2, a, y, x, b]

	

	def _monitor_controller(self):
		while True:
			try:
				events = get_gamepad()
				for event in events:
					if event.code == 'ABS_Y':
						self.LeftJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
					elif event.code == 'ABS_X':
						self.LeftJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
					elif event.code == 'ABS_RY':
						self.RightJoystickY = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
					elif event.code == 'ABS_RX':
						self.RightJoystickX = event.state / XboxController.MAX_JOY_VAL # normalize between -1 and 1
					elif event.code == 'ABS_Z':
						self.LeftTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
					elif event.code == 'ABS_RZ':
						self.RightTrigger = event.state / XboxController.MAX_TRIG_VAL # normalize between 0 and 1
					elif event.code == 'BTN_TL':
						self.LeftBumper = event.state
					elif event.code == 'BTN_TR':
						self.RightBumper = event.state
					elif event.code == 'BTN_SOUTH':
						self.A = event.state
					elif event.code == 'BTN_NORTH':
						self.X = event.state
					elif event.code == 'BTN_WEST':
						self.Y = event.state
					elif event.code == 'BTN_EAST':
						self.B = event.state
					elif event.code == 'BTN_THUMBL':
						self.LeftThumb = event.state
					elif event.code == 'BTN_THUMBR':
						self.RightThumb = event.state
					elif event.code == 'BTN_SELECT':
						self.Back = event.state
					elif event.code == 'BTN_START':
						self.Start = event.state
					elif event.code == 'BTN_TRIGGER_HAPPY1':
						self.LeftDPad = event.state
					elif event.code == 'BTN_TRIGGER_HAPPY2':
						self.RightDPad = event.state
					elif event.code == 'BTN_TRIGGER_HAPPY3':
						self.UpDPad = event.state
					elif event.code == 'BTN_TRIGGER_HAPPY4':
						self.DownDPad = event.state
			except: pass


def initialize_can():
	"""
	Set up the can bus interface and send a homing message to home the steering wheel.
	"""
	#system("ip link set can0 type can bitrate 500000")
	#system("ip link set can0 up")
	
	sudoPassword = 'wijgaanwinnen22'
	command = 'sudo ip link set can0 type can bitrate 500000'
	system('echo %s|sudo -S %s' % (sudoPassword, command))
	command = 'sudo ip link set can0 up'
	system('echo %s|sudo -S %s' % (sudoPassword, command))
	
	
	bus = can.Bus(interface='socketcan', channel='can0', bitrate=500000)
	
	acc_msg = can.Message(arbitration_id=0x120, is_extended_id=True, data=[0, 0, 1, 0, 0, 0, 0, 0])
	acc_task = bus.send(acc_msg)

	sleep(0.10)
	brk_msg = can.Message(arbitration_id=0x126, is_extended_id=True, data=[0, 0, 0, 0, 0, 0, 0, 0])
	brk_task = bus.send(brk_msg)
	
	homing_msg = can.Message(arbitration_id=0x6f1, data=[0, 0, 0, 0, 0, 0, 0, 0])
	bus.send(homing_msg)
	return bus
	


if __name__ == '__main__':
	joy = XboxController()
	joy.vibrate(500)
	
	started = False
	start_time = 0
	prev_a = 0
	while not started:
		t, b, s, a, y, x, b = joy.read()
		if a == 1 and prev_a == 0:
			start_time = datetime.now()
		elif a == 1 and (datetime.now() - start_time).total_seconds() > 1.5:
			started = True
		
		prev_a = a
		sleep(.040)
		
	joy.vibrate(1000)
	bus = initialize_can()
	
	setup_completed = True

	#sending steering messages
	steering_msg = can.Message(arbitration_id=0x12c, data=[0, 0, 0, 0, 0, 0, 0, 0])
	steering_task = bus.send_periodic(steering_msg, CAN_MSG_SENDING_SPEED)
	#steering_task.stop()
	
	acc_msg = can.Message(arbitration_id=0x120, is_extended_id=True, data=[99, 0, 1, 0, 0, 0, 0, 0])
	acc_task = bus.send_periodic(acc_msg, CAN_MSG_SENDING_SPEED)
	#acc_task.stop()
	
	switch_sequence_time = 0
	switch_sequence_counter = 0
	
	active = True
	has_braked = False
	prev_a = 0
	direction = 1
	direction_time = datetime.now()
	
	try:
		while True:
			t, brake, s, a, y, x, b = joy.read()
		
			if a == 1 and prev_a == 0:
				switch_sequence_time = datetime.now()
				switch_sequence_counter += 1

				if switch_sequence_counter == 3:
					switch_sequence_counter = 0
					setup_completed = False
					active = not active
				
					if not active:
						acc_task.stop()
						steering_task.stop()
						print("Controls deactivated")
						sleep(0.2)
						joy.vibrate(200)
						sleep(0.3)
						joy.vibrate(200)
						sleep(0.3)
						joy.vibrate(200)
					else:
						acc_task.start()
						steering_task.start()
						sleep(0.2)
						joy.vibrate(750)
						print("Controls activated")
						
			
			if y == 1 and (datetime.now() - direction_time).total_seconds() > 2: # add speed check
				direction_time = datetime.now()
				direction = 0
				acc_msg.data = [0, 0, direction, 0, 0, 0, 0, 0]
				acc_task.modify_data(acc_msg)
				print("Changed gear to N")
			
			if b == 1 and (datetime.now() - direction_time).total_seconds() > 2: # add speed check
				direction_time = datetime.now()
				direction = 1
				acc_msg.data = [0, 0, direction, 0, 0, 0, 0, 0]
				acc_task.modify_data(acc_msg)
				print("Changed gear to D")
				
			if x == 1 and (datetime.now() - direction_time).total_seconds() > 2: # add speed check
				direction_time = datetime.now()
				direction = 2
				acc_msg.data = [0, 0, direction, 0, 0, 0, 0, 0]
				acc_task.modify_data(acc_msg)
				print("Changed gear to R")
			
			if switch_sequence_counter > 0 and (datetime.now() - switch_sequence_time).total_seconds() > 0.5:
				switch_sequence_counter = 0
				print("Activation / deactivation sequence aborted")
						
			prev_a = a
		
			if setup_completed == False and active:
				bus = initialize_can()
				setup_completed = True
			elif active:
				print("Throttle: {0:<3} -- Braking: {1:<3} -- Steering: {2:>8.5f}".format(joy.read()[0],joy.read()[1],joy.read()[2]), end="\n")
				acc_msg.data = [int(t), 0, direction, 0, 0, 0, 0, 0]
				acc_task.modify_data(acc_msg)
				steering_msg.data = list(bytearray(struct.pack("f", float(s))) + bytearray(4))
				steering_task.modify_data(steering_msg)
				if brake > 0 or has_braked:
					acc_task.stop()
					bus.send(can.Message(arbitration_id=0x126, is_extended_id=True, data=[brake, 0, 0, 0, 0, 0, 0, 0]))
					has_braked = brake > 0
					if has_braked == 0:
						acc_task.start()
			sleep(.040)
			
			
			
				
	except KeyboardInterrupt:
		pass
	acc_task.stop()
	#brake.stop()
	steering_task.stop()