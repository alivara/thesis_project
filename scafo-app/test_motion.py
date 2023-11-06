
import yaml
import os
from motionsensor import MotionSensor
PATH = os.path.dirname(__file__)
with open(PATH+'/config.yaml',) as file:
    conf_file = yaml.load(file, Loader=yaml.FullLoader)

MS = MotionSensor()
MS.client_id = conf_file['MQTT']['MQMES_client_id_MS']

try:
    # config the motion sensor
    MS.connect_serial('/dev/rfcomm0', 115400)
    MS.calibrate()
    MS_mode = True
except:
    print('Motion sensor is off or not paired')
    MS_mode = False

if MS_mode:
    MS.send_data = True
    MS.process_data()
    # ms_thread = Thread(target=MS.process_data)
    # ms_thread.start()