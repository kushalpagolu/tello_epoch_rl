import hid
import numpy as np
from Crypto.Cipher import AES
from datetime import datetime
import logging
from scipy.signal import butter, lfilter, welch
from collections import deque

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmotivStreamer:
    def __init__(self, buffer_size=256, fs=256):
        self.vid = 0x1234
        self.pid = 0xed02
        self.device = None
        self.cipher = None
        self.cypher_key = bytes.fromhex("31003554381037423100354838003750")
        self.logger = logging.getLogger(__name__)
        
        # EEG channel names
        self.channel_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]

        # Rolling buffers for EEG preprocessing
        self.buffer_size = buffer_size
        self.fs = fs
        self.eeg_buffers = {ch: deque(maxlen=buffer_size) for ch in self.channel_names}

    def connect(self):
        try:
            self.device = hid.device()
            self.device.open(self.vid, self.pid)

            if self.device is None:
                self.logger.error("Device object is None after opening. Check VID/PID or permissions.")
                print("Error: Device connection failed.")
                return False

            self.logger.info(f"Connected to Emotiv device {self.vid:04x}:{self.pid:04x}")
            self.device.set_nonblocking(1)
            self.cipher = AES.new(self.cypher_key, AES.MODE_ECB)
            return True

        except Exception as e:
            self.logger.error(f"Connection failed: {str(e)}")
            print(f"Error: {str(e)}")
            return False

    def disconnect(self):
        if self.device:
            self.device.close()
            self.logger.info("Disconnected from Emotiv device")

    def read_packet(self):
        """Reads and decrypts a single EEG packet."""
        try:
            encrypted = bytes(self.device.read(32))
            if not encrypted:
                return None  # No data received
            
            decrypted = self.cipher.decrypt(encrypted)
            if len(decrypted) < 32:
                self.logger.error(f"Invalid packet received. Length: {len(decrypted)}")
                return None

            # Extract data from decrypted packet
            packet = {
                'timestamp': datetime.now().isoformat(),
                'counter': decrypted[0],
                'gyro_x': decrypted[29] - 102,
                'gyro_y': decrypted[30] - 204,
                'battery': (decrypted[31] & 0x0F)
            }
            
            for i, channel_name in enumerate(self.channel_names):
                packet[channel_name] = int.from_bytes(decrypted[2*i+1:2*i+3], 'big', signed=True)
            
            return packet
        except Exception as e:
            self.logger.error(f"Error reading packet: {e}")
            print(f"Error reading packet: {e}")
            return None

    def bandpass_filter(self, data, lowcut=1.0, highcut=50.0, order=4):
        """Applies bandpass filtering to EEG signals to remove noise."""
        nyquist = 0.5 * self.fs
        low, high = lowcut / nyquist, highcut / nyquist
        b, a = butter(order, [low, high], btype='band')
        return lfilter(b, a, data)

    def compute_band_power(self, eeg_signal):
        """Computes power in Theta, Alpha, Beta, and Gamma bands."""
        if len(eeg_signal) < self.fs:  # Ensure enough data for Welch's method
            return np.zeros(4)  # Return zero band power if not enough samples
        
        freqs, psd = welch(eeg_signal, self.fs, nperseg=len(eeg_signal))
        
        bands = {
            'theta': (4, 7),
            'alpha': (8, 12),
            'beta': (13, 30),
            'gamma': (30, 50)
        }

        band_powers = [np.trapz(psd[(freqs >= f_range[0]) & (freqs <= f_range[1])]) for f_range in bands.values()]
        return np.nan_to_num(band_powers)  # Replace NaNs/Infs with 0

    def preprocess_eeg_data(self, data):
        """Processes EEG data: applies filtering, stores in buffer, and computes band power."""
        if data is None:
            print("Warning: No data received for preprocessing.")
            return np.zeros(16)  # Return zero vector if no data
        
        try:
            # Extract EEG and gyro data
            eeg_raw = np.array([data[channel] for channel in self.channel_names])
            gyro_x, gyro_y = data['gyro_x'], data['gyro_y']

            # Store EEG data in rolling buffer
            for i, channel in enumerate(self.channel_names):
                self.eeg_buffers[channel].append(eeg_raw[i])

            # Convert buffers to arrays for band power calculation
            band_power_features = np.array([
                self.compute_band_power(list(self.eeg_buffers[ch])) for ch in self.eeg_buffers.keys()
            ])

            # Flatten the band power feature vector to ensure it has 14 elements (one per channel)
            band_power_features = band_power_features.flatten()

            # Ensure that the final data has 16 elements (2 gyro + 14 band powers)
            processed_data = np.concatenate(([gyro_x, gyro_y], band_power_features))

            # Ensure the shape is correct (16,)
            if processed_data.shape != (16,):
                print(f"Warning: Processed data has incorrect shape: {processed_data.shape}")
                processed_data = np.zeros(16)  # Fallback to zero vector if shape is incorrect

            return processed_data

        except Exception as e:
            print(f"Error in preprocess_eeg_data: {e}")
            return np.zeros(16)  # Safe fallback vector
