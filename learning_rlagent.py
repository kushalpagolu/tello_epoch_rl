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
import sys

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
MODEL_FILENAME = os.path.join(os.getcwd(), "models", "drone_rl_eeg_human_loop")

#MODEL_FILENAME = "drone_rl_eeg_human_loop"
ACTION_DELAY = 5  # Seconds between actions
HUMAN_FEEDBACK_TIMEOUT = 5  # Seconds to wait for human feedback
MAX_SPEED = 30  # Maximum speed percentage

class DroneControlEnv(gym.Env):
    def __init__(self, connect_drone=False, max_speed=MAX_SPEED):
        super(DroneControlEnv, self).__init__()
        # Gyro X, Gyro Y, 14 EEG Channels
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(16,), dtype=np.float32)
        #self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)  # FB, LR
        self.action_space = spaces.Discrete(3) # 0: Move, 1: Move, 2: Land
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
            # Add any drone reset commands here
            pass
        return self.current_state

    def step(self, action):
        reward = 0
        done = False
        info = {}
        action_successful = False

        # Handle case where action might be scalar
        if isinstance(action, np.ndarray) and action.ndim == 1:
            fb, lr = action
        else:
            fb = action
            lr = 0  # Default left-right speed to 0 if scalar action is provided

        # Initialize speeds to 0 to avoid UnboundLocalError
       # forward_backward_speed = 0
      #  left_right_speed = 0

        current_time = time.time()
        if current_time - self.last_action_time >= ACTION_DELAY:
            self.last_action_time = current_time

            if not self.has_taken_off:
                # First action should be takeoff
                if self.connect_drone and self.drone_connected:
                    self.drone_controller.takeoff()
                    self.has_taken_off = True
                    self.logger.info("Drone takeoff action")
                    action_successful = True
                else:
                    self.logger.info("Drone not connected. Simulating takeoff.")
                    action_successful = True
            else:
                # Normal action processing
                #forward_backward_speed = int(action[0] * self.max_speed)
                #left_right_speed = int(action[1] * self.max_speed)
                forward_backward_speed = int(fb * self.max_speed)
                left_right_speed = int(lr * self.max_speed)

                if self.connect_drone and self.drone_connected:
                    self.drone_controller.set_forward_backward_speed(forward_backward_speed)
                    self.drone_controller.set_left_right_speed(left_right_speed)
                    self.drone_controller.send_rc_control()
                    self.logger.info(f"Drone action: FB={forward_backward_speed}, LR={left_right_speed}")
                    action_successful = True
                else:
                    self.logger.info(f"Drone not connected. Simulating action: FB={forward_backward_speed}, LR={left_right_speed}")
                    action_successful = True

            # Map predicted action to intuitive command
            action_description = self._map_action_to_command(action)
            self.logger.info(f"Predicted Action: {action_description}")

            # Get human feedback
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

            self.action_event.set()  # Signal action taken

        else:
            self.logger.info(f"Action delayed. Remaining time: {ACTION_DELAY - (current_time - self.last_action_time):.2f} seconds")
            reward = -0.05  # Small penalty for waiting

        # Basic "wander" prevention: if drone is idle, penalize
        if forward_backward_speed == 0 and left_right_speed == 0:
            reward -= 0.2

        return self.current_state, reward, done, info

    def _map_action_to_command(self, action):
        """Maps raw action values to intuitive drone commands."""
        # Check if action is a scalar (single value) or an array
        if isinstance(action, np.ndarray) and action.ndim > 0:
            fb, lr = action
        else:
            # If action is a scalar, treat it as 'fb' and set 'lr' to 0
            fb = action
            lr = 0

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

def train_drone_rl(connect_drone=True, max_speed=MAX_SPEED):
    logger = logging.getLogger(__name__)
    env = DroneControlEnv(connect_drone=connect_drone, max_speed=max_speed)
    model = env.load_or_create_model()
    env.connect_drone_controller()  # Connecting here

    num_episodes = 1000
    total_rewards = []  # List to track total rewards for each episode
    for episode in range(num_episodes):
        obs = env.reset()
        done = False
        episode_reward = 0

        while not done:
            eeg_data = obs  # This may change, depending on data source
            action = env.train_step(eeg_data)
            obs, reward, done, info = env.step(action)
            episode_reward += reward
            model.learn(total_timesteps=1000)  # Training happens here
        
        print(f"Episode {episode + 1}, Total Reward: {episode_reward}")
        total_rewards.append(episode_reward)  # Store total reward for this episode

            # Log total reward for the current episode
        logger.info(f"Episode {episode + 1}, Total Reward: {episode_reward}")

            # Calculate average reward every 100 episodes (or as desired)
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(total_rewards[-100:])  # Average of last 100 episodes
            logger.info(f"Average Reward over last 100 episodes: {avg_reward:.2f}")

            # Save the model after each episode
            logger.info(f"Episode {episode + 1}: Total Reward: {total_reward}")
            logger.info(f"Saving model after episode {episode + 1}")
            model.save(MODEL_FILENAME)
            logger.info(f"Model saved after episode {episode + 1}")
        # Save the model after each episode
        model.save(MODEL_FILENAME)
        print(f"Model saved after episode {episode + 1}")

    # Save the final model
    model.save(MODEL_FILENAME)
    print("Final model saved.")

    def train_step(self, eeg_data):
        state = np.array(eeg_data)
        self.update_state(state)
        action, _states = self.model.predict(state, deterministic=False)
        return action

    def get_human_feedback(self):
        """Asks for human feedback using the pynput library."""
        print(f"Approve action? (Press 'y' for yes, 'n' for no, timeout={HUMAN_FEEDBACK_TIMEOUT}s)")
        start_time = time.time()

        # Start a keyboard listener
        self.listener = keyboard.Listener(on_press=self._on_key_press)
        self.listener.start()

        # Wait for feedback or timeout
        while time.time() - start_time < HUMAN_FEEDBACK_TIMEOUT:
            if self.human_feedback is not None:
                break
            time.sleep(0.1)  # Sleep to avoid busy-waiting

        # Stop the listener
        self.listener.stop()
        self.listener.join()

        if self.human_feedback is None:
            print("No feedback received, proceeding...")
        return self.human_feedback

    def _on_key_press(self, key):
        """Callback function for keypress events."""
        try:
            if key.char == 'y':
                print("Action approved.")
                self.human_feedback = True
            elif key.char == 'n':
                print("Action rejected.")
                self.human_feedback = False
        except AttributeError:
            pass  # Ignore non-character keys

if __name__ == "__main__":
    train_drone_rl(connect_drone=True, max_speed=25)  # or False for simulation
