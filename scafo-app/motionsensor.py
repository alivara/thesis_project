import time
import os
from datetime import datetime
import subprocess
import numpy as np
import serial
import struct
import yaml
from threading import Thread
from mqtt_cl import MQTTClient

PATH = os.path.dirname(__file__)
with open(os.path.join(PATH, 'config.yaml'), 'r') as file:
    conf_file = yaml.load(file, Loader=yaml.FullLoader)

# setting the server config of MQTT
MQMES = MQTTClient(conf_file = conf_file['MQTT'],MQMES_client_id = conf_file['MQTT']['MQMES_client_id'])
MQMES.topics = [("scafo4/MotionSensor/dev000", 0), ("scafo4/MotionSensor/dev000/status", 0)]

class MotionSensor:
    def __init__(self):
        self.connected = False
        self.a = np.zeros(3) # acceleration
        self.e = np.zeros(3) # Euler angle
        self.v = np.zeros(3)  # velocity
        self.a0 = np.zeros(3)  # acceleration offset
        self.e0 = np.zeros(3)  # Euler angle offset
        self.rX = np.eye(3)    # rotation matrix components
        self.rY = np.eye(3)
        self.rZ = np.eye(3)
        self.ha = 0.75          # decay rate (exponential forgetting factor)
        self.he = 0.75
        self.hv = 0.9
        self.minimumAcc = np.array([0.01, 0.01, 0.05])  # noise thresholds (below these values assume 0)
        self.minimumAng = np.array([0.5, 0.5, 1])
        self.nCalib = 100        # how many measures to use for calibration
        self.nOut = 10           # how many measures to gather for a single output
        self.dt = 0.02           # how often to get readings
        self.ser = None
        self.send_data = False
        self.rows = np.array([dict() for _ in range(self.nOut)])
        self.msgOut = {"dataShape": {}, "rows": []}

    def connect_serial(self, port, baud):
        try:
            # connect to sensor:
            cd_thread = Thread(target=self.connect_device)
            cd_thread.start()
            time.sleep(5)
            self.ser = serial.Serial(port, baud, timeout=5)
            data = self.ser.read(1)

            # Flush initial data
            while struct.unpack('B', data[0:1])[0] != 0x55:
                data = self.ser.read(1)
        except KeyboardInterrupt:
            print("Error in sending data to MQTT")


    def get_acceleration(self):
        # We assume that we are at the beginning of a record, past the initial 0x55 header byte
        data = self.ser.read(11)
        cmd = struct.unpack('B', data[0:1])[0]
        while cmd != 0x51:
            data = self.ser.read(11)
            cmd = struct.unpack('B', data[0:1])[0]
        a = np.array(struct.unpack('<hhh', data[1:7])) / (104.490 * 2)  # m^2/s
        return a

    def get_angle(self):
        data = self.ser.read(11)
        cmd = struct.unpack('B', data[0:1])[0]
        while cmd != 0x51:
            data = self.ser.read(11)
            cmd = struct.unpack('B', data[0:1])[0]
        e = np.array(struct.unpack('<hhh', data[1:7])) / 182.044444  # deg
        return e

    def connect_device(self):
        subprocess.run(["sudo","rfcomm","connect","hci0",conf_file['Sen']['MAC']])

    def convert_numpy_to_list(self,obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, list):
            return [self.convert_numpy_to_list(item) for item in obj]
        elif isinstance(obj, dict):
            return {key: self.convert_numpy_to_list(value) for key, value in obj.items()}
        else:
            return obj
        
    def calibrate(self):
        # # Calibration loop
        # a = np.zeros(3)  # acceleration
        # e = np.zeros(3)  # Euler angle
        for n in range(self.nCalib):
            a = self.get_acceleration()
            e = self.get_angle()
            self.a0 = self.a0 + a / self.nCalib
            self.e0 = self.e0 + e / self.nCalib

    def compute_rotation(self):
        e = self.get_angle()
        self.rX[1, 1] = np.cos(e[0])
        self.rX[1, 2] = -np.sin(e[0])
        self.rX[2, 1] = np.sin(e[0])
        self.rX[2, 2] = np.cos(e[0])

        self.rY[0, 0] = np.cos(e[1])
        self.rY[0, 2] = np.sin(e[1])
        self.rY[2, 0] = -np.sin(e[1])
        self.rY[2, 2] = np.cos(e[1])

        self.rZ[0, 0] = np.cos(e[2])
        self.rZ[0, 1] = -np.sin(e[2])
        self.rZ[1, 0] = np.sin(e[2])
        self.rZ[1, 1] = np.cos(e[2])

    def process_data(self):
        # connect to MQTT to send data
        MQMES.connect()
        # define n
        n = 0

        while self.send_data:
            
            # time
            t = str(datetime.now().astimezone())
            # Update with decaying memory
            a = self.ha * (self.get_acceleration() - self.a0) + (1 - self.ha) * self.a
            e = self.he * (self.get_angle() - self.e0) + (1 - self.he) * self.e
            
            # Compute and apply rotation
            self.compute_rotation()
            a = a @ self.rZ @ self.rY @ self.rX

            # Noise thresholding
            a[np.abs(a) < self.minimumAcc] = 0
            e[np.abs(e) < self.minimumAng] = 0

            # Update using zero-order hold (alternative: midpoint) and decay
            v = self.hv * (self.v + a * self.dt) - (1 - self.hv) * self.v

            # Output
            
            t1  = time.time()
            t = str(datetime.now().astimezone())
            if n + 1 < self.nOut:
                self.rows[n]["accX"] = a[0]
                self.rows[n]["accY"] = a[1]
                self.rows[n]["accZ"] = a[2]
                self.rows[n]["angX"] = e[0]
                self.rows[n]["angY"] = e[1]
                self.rows[n]["angZ"] = e[2]
                self.rows[n]["VX"] = v[0]
                self.rows[n]["VY"] = v[1]
                self.rows[n]["VZ"] = v[2]
                self.rows[n]["speed"] = np.linalg.norm(v)
            else:
                self.msgOut["rows"] = self.rows
                MQMES.dataMsg = self.convert_numpy_to_list(self.msgOut)
                MQMES.run()
                # Reset message and counter
                n = 0
                self.rows = np.array([dict() for _ in range(self.nOut)])
                print(time.time()-t1)
            
            n += 1
            time.sleep(self.dt)


    
# if __name__ == "__main__":
#     motion_sensor = MotionSensor()
#     motion_sensor.connect_serial('/dev/rfcomm0', 115400)
#     motion_sensor.read_config('config.yaml')
#     motion_sensor.calibrate()
#     motion_sensor.process_data()
