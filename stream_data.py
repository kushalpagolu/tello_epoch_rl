import hid
import numpy as np
from Crypto.Cipher import AES
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class EmotivStreamer:
    def __init__(self):
        self.vid = 0x1234
        self.pid = 0xed02
        self.device = None
        self.cipher = None
        self.cypher_key = bytes.fromhex("31003554381037423100354838003750")
        self.data_store = []
        self.channel_names = ["AF3", "F7", "F3", "FC5", "T7", "P7", "O1", "O2", "P8", "T8", "FC6", "F4", "F8", "AF4"]
        self.logger = logging.getLogger(__name__)

    def connect(self):
        try:
            self.device = hid.device()
            self.device.open(self.vid, self.pid)

            if self.device is None:
                self.logger.error("Device object is None after opening. Check VID/PID or permissions.")
                return False

            self.logger.info(f"Connected to Emotiv device {self.vid:04x}:{self.pid:04x}")
            self.device.set_nonblocking(1)
            self.cipher = AES.new(self.cypher_key, AES.MODE_ECB)
            return True

        except Exception as e:
            self.logger.error(f"Connection failed: {str(e)}")
            return False

    def disconnect(self):
        if self.device:
            self.device.close()
            self.logger.info("Disconnected from Emotiv device")

    def read_packet(self):
        try:
            encrypted = bytes(self.device.read(32))
            if not encrypted:
                return None
            decrypted = self.cipher.decrypt(encrypted)
            if len(decrypted) < 32:
                self.logger.error(f"Invalid packet received. Length: {len(decrypted)}")
                return None
            
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
            return None

    def preprocess_eeg_data(self, data):
        if data is None:
            self.logger.error("Received None data in preprocess_eeg_data")
            return None

        try:
            # Extract EEG and gyro data
            eeg_features = np.array([data[channel] for channel in self.channel_names])
            gyro_x = data['gyro_x']
            gyro_y = data['gyro_y']

            # Combine EEG and gyro data into a single feature vector
            processed_data = np.concatenate(([gyro_x, gyro_y], eeg_features))

            return processed_data
        except Exception as e:
            self.logger.error(f"Error in preprocess_eeg_data: {str(e)}")
            return None
