from djitellopy import Tello
import time
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class TelloController:
    def __init__(self, host='192.168.10.1', port=8889):
        self.tello = Tello(host, port)
        self.connected = False
        self.forward_backward_speed = 0
        self.left_right_speed = 0
        self.up_down_speed = 0
        self.yaw_velocity = 0
        self.logger = logging.getLogger(__name__)

    def connect(self):
        try:
            self.logger.info("Connecting to Tello")
            self.tello.connect()
            #self.tello.set_loglevel(self.tello.LOG_WARN)  # Suppress non-UTF-8 messages
            self.connected = True
            self.logger.info(f"Tello Battery: {self.tello.get_battery()}%")
            return True
        except Exception as e:
            self.logger.error(f"Error connecting to Tello: {e}")
            return False

    def takeoff(self):
        try:
            self.logger.info("Taking off")
            self.tello.takeoff()
        except Exception as e:
            self.logger.error(f"Error during takeoff: {e}")

    def move_forward(self, distance):
        try:
            self.logger.info(f"Moving forward {distance} cm")
            self.tello.move_forward(distance)
        except Exception as e:
            self.logger.error(f"Error moving forward: {e}")

    def move_back(self, distance):
        try:
            self.logger.info(f"Moving back {distance} cm")
            self.tello.move_back(distance)
        except Exception as e:
            self.logger.error(f"Error moving back: {e}")

    def move_left(self, distance):
        try:
            self.logger.info(f"Moving left {distance} cm")
            self.tello.move_left(distance)
        except Exception as e:
            self.logger.error(f"Error moving left: {e}")

    def move_right(self, distance):
        try:
            self.logger.info(f"Moving right {distance} cm")
            self.tello.move_right(distance)
        except Exception as e:
            self.logger.error(f"Error moving right: {e}")

    def land(self):
        try:
            self.logger.info("Landing")
            self.tello.land()
        except Exception as e:
            self.logger.error(f"Error during landing: {e}")

    def end(self):
        try:
            self.logger.info("Ending Tello session")
            self.tello.disable_mission_pads()
            self.tello.end()
        except Exception as e:
            self.logger.error(f"Error ending Tello session: {e}")

    def set_forward_backward_speed(self, speed):
        self.forward_backward_speed = speed
        self.tello.set_forward_backward_speed(speed)

    def set_left_right_speed(self, speed):
        self.left_right_speed = speed
        self.tello.set_left_right_speed(speed)

    def set_up_down_speed(self, speed):
        self.up_down_speed = speed
        self.tello.set_up_down_speed(speed)

    def set_yaw_velocity(self, speed):
        self.yaw_velocity = speed
        self.tello.set_yaw_velocity(speed)

    def send_rc_control(self):
        try:
            self.tello.send_rc_control(self.left_right_speed, self.forward_backward_speed, self.up_down_speed, self.yaw_velocity)
        except Exception as e:
            self.logger.error(f"Error sending RC control: {e}")
