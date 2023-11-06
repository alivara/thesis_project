import os
import time
import sys
import requests
import socket
import base64
import yaml
import json
import numpy as np
from datetime import datetime
from collections import deque
from plistlib import FMT_XML
from threading import Thread
from deformationdetector import DeformationDetector
from mqtt_cl import MQTTClient
from flask import Flask, Response, render_template, request, session
from flask_socketio import SocketIO
from imagemanager import ImageManager
from scafocapture_picam2 import ScafoCapture
from motionsensor import MotionSensor
from mpu6050data import MPU6050Data
from filetransfer import USBFileManager
import logging

# path config
PATH = os.path.dirname(__file__)

# reading the config file
with open(PATH+'/config.yaml',) as file:
    conf_file = yaml.load(file, Loader=yaml.FullLoader)

print(f'configuration of the session: {conf_file}')
CAP = ScafoCapture()
DEF = DeformationDetector()
MAN = ImageManager()
MPU = MPU6050Data()
# FTPT = FTPTransfer()
USBT = USBFileManager()
MS = MotionSensor()

# setting the server config of MQTT
MQMES = MQTTClient(conf_file = conf_file['MQTT'],MQMES_client_id = conf_file['MQTT']['MQMES_client_id'])

try:
    # config the motion sensor
    MS.connect_serial('/dev/rfcomm0', 115400)
    MS.calibrate()
    MS_mode = True
except:
    print('Motion sensor is off or not paired')
    MS_mode = False

# check USB 
USBT.check_usb() # if so, go on debug mode

# exposer time camera
CAP.ExposureTime = conf_file['Camera']['ExposureTime']['value']

# Diple
shift_dipole_threshold = conf_file['Deformation']['Dipole']['threshold']
DEF.percentile_threshold = conf_file['Deformation']['Dipole']['percentile_threshold']
DEF.thresh_multiplier = conf_file['Deformation']['Dipole']['thresh_multiplier']
# kmean
DEF.num_k = conf_file['Deformation']['kmean']['num_cluster']
# ROI
DEF.pixel_increase = conf_file['Deformation']['ROI']['increase_pixel']
# Flask app
app = Flask(__name__,static_url_path=PATH+conf_file['Flask']['path'])
app.secret_key = b'_5#y2L"F4Q8z\n\xec]/' # session secret (should be changed)
socketio = SocketIO(app,async_mode='threading')


##############------------------- TEST
# DEF.frame_pos_px = ([2784 ,3317], [3468 ,3341], [3464 ,3998], [2732 ,3946])
# DEF.dipoles_base = {1: ((176, 137), (118, 105), 66.24198064671678), 
#                     2: ((571, 155), (628, 113), 70.80254232723568), 
#                     3: ((146, 495), (114, 528), 45.967379738244816), 
#                     4: ((612, 513), (634, 561), 52.80151512977634)}
# DEF.d_kernel = 5
##############------------------- END
#########################################################
# region Common
def setup_logging(level=logging.INFO):
    app.logger.setLevel(level)
    # Here we define our formatter
    FORMAT = "[  %(asctime)s ] : ----- %(message)s ----- "
    formatter = logging.Formatter(FORMAT)
    consoleHandler = logging.StreamHandler(stream=sys.stdout)
    consoleHandler.setFormatter(formatter)
    consoleHandler.setLevel(level)
    app.logger.handlers.clear()
    app.logger.addHandler(consoleHandler)

setup_logging()

def init_session():
    if 'sessionid' not in session:
        session['sessionid'] = str(time.time()).replace('.', '_')
    # conf_file['Flask']['sessionid'] = session['sessionid']
    # MAN.filename_prefix = session['sessionid']

@app.route('/')
def index():
    init_session()
    print(session)
    print('prefix: ', MAN.filename_prefix)
    
    return render_template('dashboard.html')

@app.route('/video_feed')
def video_feed():
    return Response(CAP.start_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

 
# endregion
#########################################################

#########################################################
# region Setup

@app.route('/setup')
def setup():
    init_session()
    # if conf_file['Mosaico'] and conf_file['Camera']['ExposureTime']['change']:
    #     # return render_template('setup.html',mosaico=conf_file['Mosaico'],exp_val = CAP.ExposureTime ,exp_time=conf_file['Camera']['base_conf']['ExposureTime']['change'])
    #     return render_template('setup.html',mosaico=conf_file['Mosaico'],exp_val = CAP.ExposureTime ,exp_time=conf_file['Camera']['ExposureTime']['change'])
    # elif conf_file['Mosaico']:
    #     return render_template('setup.html',mosaico=conf_file['Mosaico'])
    # else:
    #     return render_template('setup.html')
    return render_template('setup.html',mosaico=conf_file['Mosaico'],exp_val = CAP.ExposureTime ,exp_time=conf_file['Camera']['ExposureTime']['change'])
    

@app.route('/detect_roi', methods=['POST'])
def detect_roi():
    # make directory for current process
    USBT.make_directory_on_usb(str(datetime.today().strftime("%Y%m%d-%H-%M-%S")))
    # do the process of ROI
    capture_time = datetime.now().strftime("%H-%M-%S")
    frame_img_raw = CAP.capture_n() # capture_single
    frame_mask_img, frame_detect_img = DEF.get_frame_tags(frame_img_raw)
    # frame_img_raw_path = MAN.store_image(frame_img_raw, 'frame_img_raw')

    app.logger.info('Focus camera on the ROI')
    CAP.focus_mode(DEF.frame_pos_px,DEF.fo_box)

    frame_img_raw = CAP.capture_n() # capture_single
    frame_img_raw_path = MAN.store_image(frame_img_raw, 'frame_img_raw')

    frame_mask_img, frame_detect_img = DEF.get_frame_tags(frame_img_raw)
    frame_mask_pres = DEF.perspective_transform(frame_mask_img)
    frame_img_pres = DEF.perspective_transform(frame_img_raw)
    MAN.store_image(frame_img_pres, 'frame_img_pres')
    # frame_detect_img_path = MAN.store_image(frame_detect_img, 'frame_detect_img')
    frame_mask_pres_path = MAN.store_image(frame_mask_pres, 'frame_mask_pres')

    if USBT.debug_mode:
        app.logger.info('Transefer ROI image to USB')
        USBT.transfer_file(frame_img_raw_path,"frame_img_raw"+str(capture_time)+'.png')
   
    
    

    return frame_mask_pres_path

@app.route('/detect_dipole', methods=['POST'])
def detect_dipole():
    app.logger.info('Capture base dipoles')
    frame_img_pres = MAN.get_image('frame_img_pres')
    DEF.n_dipole = int(request.form['n_dipole'])
    DEF.dipole_detector_02(frame_img_pres,'base') # change the number
    app.logger.info('Showing results')
    base_img_raw_path = MAN.get_output_path('frame_img_pres', dir='')
    base_dipole_img_path = MAN.get_output_path('base_dipole_img', dir='')
    n_dipole = int(list(DEF.dipoles_base.keys())[-1])
    
    return f'Number Of Detected Dipoles: {n_dipole} ; {base_img_raw_path},{base_dipole_img_path}'
   

# Mosaico Configuration
@app.route('/project_pattern', methods=['POST'])
def project_pattern():
    # read the numbers from web
    br_str = request.form['br']
    ro_str = request.form['ro']
    fo_str = request.form['fo']
    zo_str = request.form['zo']
    
    # run the command in terminal 
    os.system("sudo ola_set_dmx --universe 0 --dmx 255,"+str(br_str)+",10,10,26,"+str(ro_str)+",0,0,"+str(fo_str)+","+str(zo_str))
    # # Turn off
    # os.system("sudo ola_set_dmx --universe 0 --dmx 0,0,0,0,0,0,0,0,0,0")
    return "Change Submitted"


@app.route('/capture_baseline', methods=['POST'])
def capture_baseline():
    app.logger.info('Capture base image')
    capture_time = datetime.now().strftime("%H-%M-%S")
    base_img_raw  = CAP.capture_n() # capture_single
    path_img_base_path = MAN.store_image(base_img_raw, 'base_img_raw')
    base_img = DEF.perspective_transform(base_img_raw)
    base_img_path = MAN.store_image(base_img, 'base_img')
    DEF.dipole_detector_02(base_img,'base') 
    app.logger.info('Showing results')
    # the jaccard 
    if USBT.debug_mode:
        app.logger.info('Transefer Base image to USB')
        USBT.transfer_file(path_img_base_path,"base_img_raw"+str(capture_time)+'.png')

    return base_img_path

# endregion
#########################################################

#########################################################
# region Monitoring

@app.route('/monitoring')
def monitoring():
    return render_template('monitoring.html')

@app.route('/stop_monitoring', methods=['POST'])
def stop_monitoring():
    global monitoring_thread
    app.logger.info('Stopping monitoring thread')
    if MS_mode:
        print('None')
        # MS.send_data = False

    MPU.cap_status = False
    CAP.monitoring = False # stopping the monitoring loop
    monitoring_thread.join()
    status = 'Monitoring Stopped'
    socketio.emit('monitor_status', status)
    return 'STOP Monitoring...'

@app.route('/start_monitoring', methods=['POST'])
def start_monitoring():
    # motion sensor
    if MS_mode:
        print('None')
        MS.send_data = True
        ms_thread = Thread(target=MS.process_data)
        ms_thread.start()
    global monitoring_thread
    monitoring_thread = Thread(target=monitoring_thread, args=(request,))
    app.logger.info('Starting monitoring thread')
    monitoring_thread.start()
    status = 'Monitoring Started'
    socketio.emit('monitor_status', status)
    monitoring_thread.join()
    return 'Monitoring...'

# Convert NumPy arrays inside the dictionary to lists
def convert_numpy_to_list(obj):
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, list):
        return [convert_numpy_to_list(item) for item in obj]
    elif isinstance(obj, dict):
        return {key: convert_numpy_to_list(value) for key, value in obj.items()}
    else:
        return obj

def monitoring_thread(request):
    '''
    the threads are:
        1.
        2. 
    '''
    # running MPU sensor to capture data
    app.logger.info('capture acc data')
    MPU.cap_status = True
    acc_thread = Thread(target=MPU.run)
    acc_thread.start()
    
    # connecting to the MQTT server
    app.logger.info('Connect to MQTT')
    MQMES.connect() # on after debug
    
    # re-Config the camera 
    app.logger.info('re-Config the camera')
    CAP.monitoring = True
    CAP.stop_stream()
    CAP.initialize_video_capture()
    CAP.continue_stream()
    
    app.logger.info('Getting Base Image')
    base_img = MAN.get_image('base_img')
    # base_img = MAN.get_image('frame_img_pres')
    
    
    
    while CAP.monitoring:
        # sending data from motion sensor to 
        
        status = 'Evaluation in progress'
        socketio.emit('monitor_status', status)
        # socketio.emit('monitor_delay_elapsed')
        
        capture_time = datetime.now().strftime("%H-%M-%S")
        app.logger.info(f'Capturing @ {capture_time}')
        mon_img_raw = CAP.capture_n() # capture_single
        mon_img_path = MAN.store_image(mon_img_raw, 'mon_img_raw') 
        mon_img = DEF.perspective_transform(mon_img_raw)
        MAN.store_image(mon_img, 'mon_img') 
        
        app.logger.info('Continue Streaming')
        CAP.continue_stream()
        
        app.logger.info('Doing process in threads')
        # build shifting results
        result = {
            "capture_time" : capture_time,
            "bumps_detected": False,
            "shifting_detected" : False}
        
        # this part need to be updated for having monitonig in same part
        # here has been commented to avoid shiftting detection
        
        # the jaccard 
        if USBT.debug_mode:
            app.logger.info('Transefer mon image to USB')
            thread_mon_img_FTP = Thread(target=USBT.transfer_file, args=(mon_img_path,str(capture_time)+'.png'))
            thread_mon_img_FTP.start()
        
        # finding the dipoles
        app.logger.info('Dipoles Proccessing ')
        # led_positions, _ , _ = DEF.detect_led_tags(mon_img, 'green', int(list(DEF.dipoles_base.keys())[-1]))
        DEF.dipole_detector_02(mon_img,'mon') # change the number
        thread_dipole = Thread(target=DEF.dipole_shifting, args=(base_img, mon_img,DEF.positions_blue)) # change number
        thread_dipole.start()
        
        # the jaccard 
        # tol_jc_thres = 0.01
        app.logger.info('Jaccard Proccessing')
        thread_jac = Thread(target=DEF.dif_jac_val, args=(mon_img, base_img, conf_file['Deformation']['Jaccard']['threshold']))
        thread_jac.start()

        # Wait for the thread to complete
        app.logger.info('Wait for Jaccard Proccessing')
        thread_jac.join()
        app.logger.info('Wait for Dipole Proccessing')
        thread_dipole.join()
        if USBT.debug_mode:
            app.logger.info('Wait for mon FTP Transfer')
            thread_mon_img_FTP.join()
        
        # build bumps results for jaccard
        def_detection = {}
        print(f'jac con {DEF.jac_con}')
        if not DEF.jac_con: # change this with jaccard condition
            # MQTT MSG
            def_detection['threshold'] = conf_file['Deformation']['Jaccard']['threshold'] # sending shifting results to MQTT
            def_detection['status'] = '1'
            # WEB MSG
            result["bumps_detected"] = True
            result["jac_plot_path"] = MAN.get_output_path('jac_plot', dir='') 
            if USBT.debug_mode:
                app.logger.info('Transefer mon image to USB')
                thread_jc_img_FTP = Thread(target=USBT.transfer_file, args=(result["jac_plot_path"],"jac_plot_path"+str(capture_time)+'.png'))
                thread_jc_img_FTP.start()
                app.logger.info('Wait for jac_plot USB Transfer')
                thread_jc_img_FTP.join()
            # converting the image to be sent by MQTT
            with open(result["jac_plot_path"], "rb") as image_file:
                encoded_string = base64.b64encode(image_file.read())

            def_detection["Deformation Plot"] = [encoded_string.decode('utf-8')]
            MQMES.dataMsg["Deformation_results"] = def_detection

        else:
            def_detection['threshold'] = conf_file['Deformation']['Jaccard']['threshold'] # sending shifting results to MQTT
            def_detection['status'] = '0'
            MQMES.dataMsg["Deformation_results"] = def_detection
            

        # Specify the threshold value
        shifting_results = []
        shifting_msg_dip = {}
        for key_base, (_, _, closest_distance) in DEF.dipoles_shiffted.items():
            if closest_distance > shift_dipole_threshold:
                shifting_results.append(f'Detected shifting of led {str(key_base)} by {round(closest_distance)} mm')
                shifting_msg_dip[str(key_base)] = str(closest_distance)
                
        shifting_msg = {}      
        if len(shifting_results) != 0:
            shifting_msg['threshold'] = shift_dipole_threshold # sending shifting results to MQTT
            shifting_msg['status'] = '1'
            shifting_msg['Dipole_base_info'] = DEF.dipoles_base
            result["shifting_detected"] = True
            result["shifting_results"] = shifting_results
            MQMES.dataMsg["shifting_results"] = shifting_msg
            MQMES.dataMsg["Shifted_dipoles"] = shifting_msg_dip
            result["shifting_image_path"] = MAN.get_output_path('shift_img', dir='')
        else:
            shifting_msg['threshold'] = shift_dipole_threshold # sending shifting results to MQTT
            shifting_msg['status'] = '0'
            shifting_msg['Dipole_base_info'] = DEF.dipoles_base
            MQMES.dataMsg["shifting_results"] = shifting_msg
            app.logger.info("Did not detect shifting")
        
        # Sending message to the MQTT server:
        MQMES.dataMsg["ROI"] = [
                        {
                            "top_left": DEF.frame_pos_px[0], 	# pixel position	
                            "top_right": DEF.frame_pos_px[1], 	# pixel position	
                            "down_left": DEF.frame_pos_px[2], 	# pixel position	
                            "down_right": DEF.frame_pos_px[3] 	# pixel position	
                        }
                        #... repeat for all displaced leds ...
                    ]  
        
        app.logger.info('send data to the MQTT server')
        MQMES.dataMsg = convert_numpy_to_list(MQMES.dataMsg)
        MQMES.run() 
        app.logger.info('end data to the MQTT server')
        # save this message
        
        
        if USBT.debug_mode:
            # Convert the dictionary to JSON format
            mqttmsg_file = MAN.output+"/MQTTmsg"+str(capture_time)+".json"
            j_MQTT = json.dumps(convert_numpy_to_list(MQMES.dataMsg), indent=4)
            # Save the JSON data to a file
            with open(mqttmsg_file, "w") as json_file:
                json_file.write(j_MQTT)

            app.logger.info('Transfer MQTT msg to USB')
            thread_MQTTmsg = Thread(target=USBT.transfer_file, args=(mqttmsg_file,"MQTTmsg"+str(capture_time)+".json"))
            thread_MQTTmsg.start()
            app.logger.info('Wait for jac_plot FTP Transfer')
            thread_MQTTmsg.join()
            
        
        MAN.output
        MPU.out_dir = MAN.output
        MPU.transfer = True

        if USBT.debug_mode:
            app.logger.info('Transefer base image to USB')
            while MPU.saved == None:
                app.logger.info('Wait 1 second to save the CSV file')
                time.sleep(1)
            USBT.transfer_file(MPU.saved,MPU.file_name)
        
        
        socketio.emit('monitor_executed', result)
        time.sleep(CAP.monitoring_delay)
    
    status = 'Monitoring Stopped : Exiting monitoring thread'
    socketio.emit('monitor_status', status)
    app.logger.info('Exiting monitoring thread')
    
 
# endregion
#########################################################
#########################################################
# the rest will be removed in next version ...
#########################################################
# region Bumps Detection 

## Uncomment for testing 
# DEF.frame_pos_px =  ((1738,1237), (2990, 1285), (2893, 2103), (1665, 2008))
# DEF.frame_dist_px = (1684, 1300, 1675, 1267)
# DEF.horizontal_ratio = 24.69
# DEF.vertical_ratio = 24.7
# DEF.shifting_tags_pos_px = ((905, 1087), (1999, 1125), (1954, 1929), (892, 1936))
# DEF.noise_T = 4
# DEF.blur_T = 60
##

@app.route('/noise_evaluation', methods=['POST'])
def noise_evaluation():
    succ_img_raw  = CAP.capture_n()
    MAN.store_image(succ_img_raw, 'succ_img_raw')

    succ_img = DEF.perspective_transform(succ_img_raw)
    MAN.store_image(succ_img, 'succ_img')

    base_img = MAN.get_image('base_img')

    t_plot_img = DEF.threshold_evaluation(base_img, succ_img)
    t_plot_img_path = MAN.store_image(t_plot_img, 't_plot_img')
    
    return t_plot_img_path

@app.route('/store_T', methods=['POST'])
def store_T():
    T_str = request.form['T']
    DEF.noise_T = int(T_str) if T_str.isdigit() else 0
    return 'Noise threshold stored'


# endregion
#########################################################

if __name__ == "__main__":
    socketio.run(app,host=conf_file['Flask']['host'],port=conf_file['Flask']['port'])


# session data is now replaced by object reference DeformationDetector
# use the below code to serialize, store in session, deserialize data

# # serialize and store in session
# session['frame_pos_px'] = serialize_numpy(frame_pos_px)
# session['frame_dist_px'] = serialize_numpy(frame_dist_px)

# def serialize_numpy(obj):
#     # https://quick-adviser.com/are-numpy-arrays-json-serializable/
#     nplist = np.array(obj).tolist()
#     json_str = json.dumps(nplist)
#     return json_str


# def deserialze_numpy(str):
#     json_load = json.loads(str)
#     restored = np.asarray(json_load)
#     return restored

