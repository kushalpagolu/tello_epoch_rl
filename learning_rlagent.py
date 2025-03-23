import gym
from stable_baselines3 import PPO
import numpy as np
from drone_control import TelloController
from gym import spaces
import os
import logging
import time
import threading
from pynput import keyboard  # Import the pynput library

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

MODEL_FILENAME = "drone_rl_eeg_human_loop"
ACTION_DELAY = 5  # Seconds between actions
HUMAN_FEEDBACK_TIMEOUT = 5  # Seconds to wait for human feedback
MAX_SPEED = 30  # Maximum speed percentage

class DroneControlEnv(gym.Env):
    def __init__(self, connect_drone=False, max_speed=MAX_SPEED):
        super(DroneControlEnv, self).__init__()
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        self.action_space = spaces.Discrete(3)  # 0: Move, 1: Move, 2: Land
        self.current_state = np.zeros(16)
        self.connect_drone = connect_drone
        self.drone_controller = None  # Initialize to None
        if connect_drone:
            self.drone_controller = TelloController()
        self.drone_connected = False
        self.model = None
        self.logger = logging.getLogger(__name__)
        self.last_action_time = 0
        self.action_event = threading.Event()  # To signal action taken
        self.human_feedback = None
        self.max_speed = max_speed  # Set max speed
        self.has_taken_off = False  # Flag to track if the drone has taken off
        self.listener = None  # Keyboard listener

    def connect_drone_controller(self):
        if self.connect_drone and not self.drone_connected:
            self.drone_connected = self.drone_controller.connect()
            if self.drone_connected:
                self.logger.info("Drone connected successfully")
            else:
                self.logger.error("Failed to connect to drone")
        return self.drone_connected

    def reset(self):
        self.current_state = np.zeros(16)  # Reset to zeros
        if self.connect_drone and self.drone_connected:
            pass
        return self.current_state

    def step(self, action):
        reward = 0
        done = False
        info = {}
        action_successful = False

        forward_backward_speed = 0
        left_right_speed = 0

        current_time = time.time()
        if current_time - self.last_action_time >= ACTION_DELAY:
            self.last_action_time = current_time

            if not self.has_taken_off:
                if self.connect_drone and self.drone_connected:
                    self.drone_controller.takeoff()
                    self.has_taken_off = True
                    self.logger.info("Drone takeoff action")
                    action_successful = True
                else:
                    self.logger.info("Drone not connected. Simulating takeoff.")
                    action_successful = True
            else:
                if action == 2:  # Land action
                    if self.connect_drone and self.drone_connected:
                        self.drone_controller.land()  # Land the drone
                        self.logger.info("Drone landing action")
                        action_successful = True
                    else:
                        self.logger.info("Drone not connected. Simulating landing.")
                        action_successful = True
                else:
                    forward_backward_speed = int(action[0] * self.max_speed)
                    left_right_speed = int(action[1] * self.max_speed)

                    if self.connect_drone and self.drone_connected:
                        self.drone_controller.set_forward_backward_speed(forward_backward_speed)
                        self.drone_controller.set_left_right_speed(left_right_speed)
                        self.drone_controller.send_rc_control()
                        self.logger.info(f"Drone action: FB={forward_backward_speed}, LR={left_right_speed}")
                        action_successful = True
                    else:
                        self.logger.info(f"Drone not connected. Simulating action: FB={forward_backward_speed}, LR={left_right_speed}")
                        action_successful = True

            action_description = self._map_action_to_command(action)
            self.logger.info(f"Predicted Action: {action_description}")

            self.human_feedback = self.get_human_feedback()

            if self.human_feedback is not None:
                if self.human_feedback:
                    reward = 1  # Positive feedback
                else:
                    reward = -1  # Negative feedback
                    if self.connect_drone and self.drone_connected:
                        self.drone_controller.set_forward_backward_speed(-forward_backward_speed)
                        self.drone_controller.set_left_right_speed(-left_right_speed)
                        self.drone_controller.send_rc_control()
                        self.logger.info("Action counteracted due to negative feedback")
            else:
                reward = 0.1  # Small reward for taking action (no feedback)

            self.action_event.set()

        else:
            self.logger.info(f"Action delayed. Remaining time: {ACTION_DELAY - (current_time - self.last_action_time):.2f} seconds")
            reward = -0.05  # Small penalty for waiting

        if forward_backward_speed == 0 and left_right_speed == 0:
            reward -= 0.2

        return self.current_state, reward, done, info

    def _map_action_to_command(self, action):
        """Maps raw action values to intuitive drone commands."""
        if action == 2:
            return "Land"
        fb, lr = action
        if fb > 0.5:
            return "Move Forward"
        elif fb < -0.5:
            return "Move Backward"
        elif lr > 0.5:
            return "Move Right"
        elif lr < -0.5:
            return "Move Left"
        else:
            return "Hover"

    def update_state(self, new_state):
        self.current_state = new_state

    def load_or_create_model(self):
        if os.path.exists(f"{MODEL_FILENAME}.zip"):
            try:
                self.model = PPO.load(MODEL_FILENAME, env=self)
                self.logger.info("Loaded existing model")
            except Exception as e:
                self.logger.error(f"Error loading existing model: {e}. Creating a new model instead.")
                self.model = PPO("MlpPolicy", self, verbose=1)
                self.logger.info("Created new model")
        else:
            self.model = PPO("MlpPolicy", self, verbose=1)
            self.logger.info("Created new model")
        return self.model
