import threading
from datetime import datetime
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from visualizer_realtime import RealtimeEEGVisualizer
from stream_data_emotivdisconnect import EmotivStreamer
from learning_rlagent_emotivdisconnect import DroneControlEnv, PPO
import time
import os
import logging
import signal
import argparse

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

stop_saving_thread = threading.Event()
stop_main_loop = threading.Event()
data_store = []  # Initialize data store

def save_data_continuously(data_store, filename_prefix="eeg_gyro"):
    while not stop_saving_thread.is_set():
        if data_store:
            filename = os.path.join("data", f"{filename_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
            try:
                df = pd.DataFrame(data_store)
                df.to_excel(filename, index=False)
                logger.info(f"Data saved to {filename}")
                data_store.clear()
            except Exception as e:
                logger.error(f"Error saving data to Excel: {str(e)}")
        time.sleep(10)

def process_data(emotiv, visualizer, env, model, connect_drone):
    consecutive_empty_packets = 0
    max_empty_packets = 200  # Increased tolerance for empty packets

    while not stop_main_loop.is_set():
        packet = emotiv.read_packet()
        
        if packet is None:
            consecutive_empty_packets += 1
            if consecutive_empty_packets > max_empty_packets:
                logger.error("Too many consecutive empty packets. Reconnecting...")
                emotiv.disconnect()
                time.sleep(5)  # Added reconnection delay
                if emotiv.connect():
                    consecutive_empty_packets = 0
                else:
                    logger.error("Failed to reconnect. Exiting.")
                    break
            time.sleep(0.05)
            continue

        consecutive_empty_packets = 0

        # Update visualizer
        for i, channel_name in enumerate(emotiv.channel_names):
            visualizer.data_buffers[i].append(packet[channel_name])
        visualizer.update_gyro_data(packet['gyro_x'], packet['gyro_y'])

        # Process data for RL agent
        processed_data = emotiv.preprocess_eeg_data(packet)
        if processed_data is None:
            logger.warning("Failed to process EEG data. Skipping this packet.")
            continue

        env.update_state(processed_data)

        # RL agent step
        action, _states = model.predict(processed_data, deterministic=False)
        print(f"RL Agent Suggested Action: {action}")

        if connect_drone:
            # Take action with drone
            env.step(action)
        else:
            print("Drone not connected. Simulating action.")

        # Store data
        global data_store
        data_store.append(packet)  # Append packet to data store

def start_visualization(visualizer, emotiv):
    if emotiv.device is None:
        logger.warning("Emotiv device not connected. Skipping visualization.")
        return
    ani = FuncAnimation(
        visualizer.fig,
        visualizer.update,
        interval=50,
        cache_frame_data=False
    )
    plt.show()

def signal_handler(sig, frame):
    logger.info("Ctrl+C detected. Shutting down...")
    stop_saving_thread.set()
    stop_main_loop.set()
    if emotiv.device:
        emotiv.disconnect()
    
    # Save remaining data before exiting
    if data_store:
        filename = os.path.join("data", f"eeg_gyro_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        try:
            df = pd.DataFrame(data_store)
            df.to_excel(filename, index=False)
            logger.info(f"Data saved to {filename}")
        except Exception as e:
            logger.error(f"Error saving data to Excel: {str(e)}")
    
    plt.close('all')
    exit(0)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="EEG Data Streamer and Drone Control")
    parser.add_argument("--connect-drone", action="store_true", help="Connect to the drone")
    args = parser.parse_args()

    signal.signal(signal.SIGINT, signal_handler)

    emotiv = EmotivStreamer()
    visualizer = RealtimeEEGVisualizer()

    env = DroneControlEnv(connect_drone=args.connect_drone)
    model = env.load_or_create_model()

    if emotiv.connect():
        logger.info("Emotiv EEG device connected. Starting real-time EEG streaming.")
        
        if args.connect_drone:
            if env.connect_drone():
                logger.info("Drone connected successfully.")
            else:
                logger.error("Failed to connect to drone. Exiting.")
                exit(1)
        else:
            logger.info("Running in streamer mode without drone connection.")

        try:
            # Start background thread for data saving
            save_thread = threading.Thread(target=save_data_continuously, args=(data_store,))  # Pass data_store
            save_thread.daemon = True
            save_thread.start()

            # Start background thread for data processing and RL agent
            process_thread = threading.Thread(target=process_data, args=(emotiv, visualizer, env, model, args.connect_drone))
            process_thread.daemon = True
            process_thread.start()

            # Start visualization in the main thread
            start_visualization(visualizer, emotiv)

        except Exception as e:
            logger.error(f"An error occurred: {e}")
        finally:
            stop_saving_thread.set()
            stop_main_loop.set()
            time.sleep(1)
            save_thread.join(timeout=5)
            process_thread.join(timeout=5)
            if emotiv.device:
                emotiv.disconnect()
            logger.info("Session terminated.")
    else:
        logger.error("Failed to connect to Emotiv device.")
