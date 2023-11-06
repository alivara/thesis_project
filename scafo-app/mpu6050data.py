import smbus
from time import sleep
import pandas as pd
import datetime

class MPU6050Data:
    def __init__(self):
        # prepping for visualization
        self.col = ['timestamp','accel-x','accel-y','accel-z','gyro-x','gyro-y','gyro-z']
        self.out_dir = None
        self.saved = None
        self.file_name = None
        # status
        self.cap_status = False
        self.transfer = False
        # Initialize I2C bus and display
        self.bus = smbus.SMBus(3)  # Use the appropriate I2C bus number for your setup
        self.Device_Address = 0x68 # MPU6050 device address
        
        # Some MPU6050 Registers and their Addresses
        self.PWR_MGMT_1 = 0x6B
        self.SMPLRT_DIV = 0x19
        self.CONFIG = 0x1A
        self.GYRO_CONFIG = 0x1B
        self.INT_ENABLE = 0x38
        self.ACCEL_XOUT_H = 0x3B
        self.ACCEL_YOUT_H = 0x3D
        self.ACCEL_ZOUT_H = 0x3F
        self.GYRO_XOUT_H = 0x43
        self.GYRO_YOUT_H = 0x45
        self.GYRO_ZOUT_H = 0x47

        # Sensitivity
        self.acc_sen = 2048.0  # Sensitivity of 16384, 8192, 4096, or 2048 LSBs per g
        self.gyro_sen = 16.4  # Sensitivity of 131, 65.5, 32.8, or 16.4 LSBs per dps
        
    def initialize(self):
        self.bus.write_byte_data(self.Device_Address, self.SMPLRT_DIV, 7)
        self.bus.write_byte_data(self.Device_Address, self.PWR_MGMT_1, 1)
        self.bus.write_byte_data(self.Device_Address, self.CONFIG, 0)
        self.bus.write_byte_data(self.Device_Address, self.GYRO_CONFIG, 24)
        self.bus.write_byte_data(self.Device_Address, self.INT_ENABLE, 1)

    def read_raw_data(self, addr):
        #Accelero and Gyro value are 16-bit
        high = self.bus.read_byte_data(self.Device_Address, addr)
        low = self.bus.read_byte_data(self.Device_Address, addr + 1)
        
        value = ((high << 8) | low) #concatenate higher and lower value
        if value > 32768: #to get signed value from mpu6050
            value = value - 65536
            
        return value

    def run(self):
        self.initialize()
        print("Reading Data of Gyroscope and Accelerometer")
        mpu6050_vec = []
        while self.cap_status:
            # Read Accelerometer raw value
            acc_x = self.read_raw_data(self.ACCEL_XOUT_H)
            acc_y = self.read_raw_data(self.ACCEL_YOUT_H)
            acc_z = self.read_raw_data(self.ACCEL_ZOUT_H)

            # Read Gyroscope raw value
            gyro_x = self.read_raw_data(self.GYRO_XOUT_H)
            gyro_y = self.read_raw_data(self.GYRO_YOUT_H)
            gyro_z = self.read_raw_data(self.GYRO_ZOUT_H)

            # Full scale range +/- 250 degree/C as per sensitivity scale factor
            Ax = acc_x / self.acc_sen
            Ay = acc_y / self.acc_sen
            Az = acc_z / self.acc_sen

            Gx = gyro_x / self.gyro_sen
            Gy = gyro_y / self.gyro_sen
            Gz = gyro_z / self.gyro_sen
            curr_datetime = datetime.datetime.now().strftime("%Y%m%d-%H-%M-%S")
            mpu6050_vec.append([curr_datetime,Ax,Ay,Az,Gx,Gy,Gz])
            # Printing the data
            # print("\tAx=%.2f g" % Ax,"\tAy=%.2f g" % Ay, "\tAz=%.2f g" % Az)
            sleep(.5)

            if self.transfer:
                df = pd.DataFrame(mpu6050_vec,columns=self.col)
                self.saved = str(self.out_dir+curr_datetime+'.csv')
                self.file_name = str(curr_datetime+'.csv')
                df.to_csv(self.saved)

                self.transfer = False



