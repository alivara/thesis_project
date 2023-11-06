import cv2
import numpy as np
import time
from datetime import datetime
from enum import Enum
from picamera2 import Picamera2
import libcamera
import time
from skimage import img_as_ubyte
# from PIL import Image
from deformationdetector import DeformationDetector
DEF = DeformationDetector()

class CapAvgMethod(Enum):
    AddWeighted = 1
    Max = 2
    Min = 3

class ScafoCapture:
    def __init__(self, capture_resolution = (9152, 6944), streaming_resolution =  ((2312,1736),(768, 432)),ExposureTime=20000, shot_delay=0.1, monitoring_delay=30): #exp 10000
        # based on picam 2312*1736 16mp --- (768, 432),1920×1080,2312×1736,3840×2160
        self.picam2 = Picamera2() # reading the camera
        self.shot_delay = shot_delay
        self.monitoring_delay = monitoring_delay
        self.ExposureTime = ExposureTime
        self.cam_control = { "Brightness" : 0 ,"ExposureTime" : self.ExposureTime }
        self.cam_metadata = None
        
        ## config of resulation
        self.streaming = False
        self.streaming_start_time = None
        self.monitoring = False
        self.streaming_resolution = streaming_resolution
        self.capture_resolution = capture_resolution
        self.focus_parameters = None
    def initialize_video_capture(self,AnalogueGain=1.0,ExposureTime=500000):
        # Define the config of noise
        # with self.picam2.controls as ctrl:
        #     ctrl.AnalogueGain = 1.0 # Analog gain offers higher sensitivity and less noise than using only digital gain.
        #     ctrl.ExposureTime = 500000 # check this config
        # self.camera = cv2.VideoCapture(0+cv2.CAP_ANY)
        # self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        # self.camera.set(cv2.CAP_PROP_EXPOSURE, 0) # Zero for automatic exposure
        self.streaming_config()

    def streaming_config(self):
        self.picam2.stop()
        # old one due to the resolution problems
        config = self.picam2.create_still_configuration(main={"size": self.streaming_resolution[0], "format": "RGB888"},
                                                 lores={"size": self.streaming_resolution[1], "format": "YUV420"})
        # new one with lower resolution
        # config = self.picam2.create_video_configuration({"size": self.streaming_resolution, "format": "RGB888"}) # "format": "YUV420"                     
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(2)
        self.picam2.set_controls(self.cam_control)
        
    def capture_config(self):
        self.picam2.stop()
        config = self.picam2.create_still_configuration(main={"size": self.capture_resolution , "format": 'RGB888' }) 
        self.picam2.configure(config)
        self.picam2.start()
        time.sleep(1)
        self.picam2.controls.set_controls(self.cam_control)
        # self.picam2.controls.set_controls({ "ExposureTime" : self.ExposureTime })
        time.sleep(2)
        

    def focus_mode(self,frame_pos_px,led_boxes):
        # getting frame pos
        a, b, c, d = frame_pos_px
        minWidth = min([a[0], c[0], b[0],d[0]])
        minHeight = min([a[1], c[1], b[1],d[1]])
        maxWidth = max([a[0], c[0], b[0],d[0]])
        maxHeight = max([a[1], c[1], b[1],d[1]])
        
        size = self.picam2.capture_metadata()['ScalerCrop'][2:]
        full_res = self.picam2.camera_properties['PixelArraySize']
        
        while True:
            
            self.picam2.capture_metadata()
            size = [int(s * 0.95) for s in size]
            offset = [(r - s) // 2 for r, s in zip(full_res, size)]
            # Unpack the coordinates of box a
            a_x1, a_y1, a_x2, a_y2 = offset + size
            # Check if box a is outside of box b
            if a_x1 > minWidth or a_y1 > minHeight or a_x2 < maxWidth or a_y2 < maxHeight:
                break
            
            self.picam2.set_controls({"ScalerCrop": offset + size})
            
        self.picam2.controls.set_controls({"AfMode" : libcamera.controls.AfModeEnum.Continuous, 
                                           "AfMetering" : libcamera.controls.AfMeteringEnum.Windows,  
                                           "AfWindows" : [ (minWidth,minHeight,maxWidth,maxHeight) ] } )
        time.sleep(5)
        self.cam_metadata = self.picam2.capture_metadata()
        print('re-config the streaming')
        self.cam_control["AfMode"] = libcamera.controls.AfModeEnum.Manual
        # self.cam_control["ScalerCrop"] = offset + size #if needed to zoom open this
        self.cam_control["LensPosition"] = self.cam_metadata["LensPosition"]
        # self.cam_control["Brightness"] = -0.5
        # self.cam_control["ColourGains"] = (0,0)
        self.continue_stream()
        
        # # source https://forums.raspberrypi.com/viewtopic.php?p=2075317#p2075249
   
    def start_stream(self):
        # fixing the bug of refresh
        try:
            self.picam2.close()
            self.picam2.start()
        except:
            print('There is a problem in openning camera')
            self.picam2 = Picamera2()
        
        w, h = self.streaming_resolution[1]
        self.streaming = True
        self.streaming_config()
        blank_img = np.zeros((self.streaming_resolution[1][1],self.streaming_resolution[1][0],3), np.uint8)
        _, blank_img = cv2.imencode('.jpg', blank_img)
        blank_img = blank_img.tobytes()

        start_time = datetime.now()
        self.streaming_start_time = start_time
        read_success = True

        while True:
            if read_success == False:
                time.sleep(1)
                if start_time < self.streaming_start_time:
                    time_old = start_time.strftime("%H:%M:%S")
                    time_current = self.streaming_start_time.strftime("%H:%M:%S")
                    print(f'exiting old streaming thread {time_old} < {time_current}')
                    break
                else:
                    print('reinitialize video capture for new streaming thread')
                    self.initialize_video_capture()

            if self.streaming:
                frame = self.picam2.capture_buffer(name="lores")
                ## if "format": "YUV420"
                frame = np.frombuffer(frame, np.uint8).reshape(h*3//2, w)
                frame = cv2.cvtColor(frame,cv2.COLOR_YUV2BGR_I420)
                ## if RGB888
                # frame = np.frombuffer(frame, np.uint8).reshape(h, w,3)
                _, buffer = cv2.imencode('.jpg', frame)
                frame = buffer.tobytes()
            else:
               frame =  blank_img

            yield (b'--frame\r\n Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

        
    def continue_stream(self):
        self.streaming_config()
        self.streaming = True
        self.start_stream()

    def stop_stream(self):
        self.streaming = False
        self.picam2.close()
        self.picam2 = Picamera2()

    def capture_n(self, n=5, grayscale=False, normalize=True, hue_mult=1, brightness_mult=1, avg_method=CapAvgMethod.AddWeighted,focus_mode = False):
        """Captures n consecutive images, turns them into grayscale, normalize then computes and returns the average image."""
        w, h = self.capture_resolution
        self.stop_stream()
        self.capture_config()
        time.sleep(2)
        if focus_mode:
            # -- adding ROI focus
            self.picam2.controls.set_controls( { "AfMode" : libcamera.controls.AfModeEnum.Continuous, "AfMetering" : libcamera.controls.AfMeteringEnum.Windows,  "AfWindows" : [ self.focus_parameters ] } ) 
            time.sleep(2)
        # -- 
        image_data = []
        print(f'--- capturing {n} photos, averaging method: {avg_method} ---')

        start_time = time.time()
        for i in range(5): # left 5 images for waiting to config
            frame = self.picam2.capture_buffer()
            
        for i in range(n):
            frame = self.picam2.capture_buffer()
            frame = np.frombuffer(frame, np.uint8).reshape(h, w,3)
            time.sleep(self.shot_delay)
            image_data.append(frame)
        
        end_time = time.time()
        capture_duration = round(end_time - start_time, 2)
        print(f'--- captured in {capture_duration} seconds ---')
        print(f'--- averaging now ---')
     
        avg_image = image_data[0]
        for i in range(1, len(image_data)):
            if avg_method == CapAvgMethod.AddWeighted:
                alpha = 1.0/(i + 1)
                beta = 1.0 - alpha
                avg_image = cv2.addWeighted(image_data[i], alpha, avg_image, beta, 0.0)
            elif avg_method == CapAvgMethod.Max:
                avg_image = cv2.max(avg_image, image_data[i])
            elif avg_method == CapAvgMethod.Min:
                avg_image = cv2.min(avg_image, image_data[i])
        # hsv parameters
        if hue_mult != 1 or brightness_mult != 1:
            avg_image = cv2.cvtColor(avg_image, cv2.COLOR_BGR2HSV)
            avg_image[...,1] = avg_image[...,1]*hue_mult
            avg_image[...,2] = avg_image[...,2]*brightness_mult # multiply by a factor of less than 1 to reduce the brightness
            avg_image =  cv2.cvtColor(avg_image, cv2.COLOR_HSV2BGR)
        # convert to grayscale
        if grayscale:
            avg_image = cv2.cvtColor(avg_image, cv2.COLOR_BGR2GRAY) 
        # normalize light, alpha and beta should be parametrized
        if normalize:
            avg_image = cv2.normalize(avg_image, None, alpha=30, beta=200, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # convert color channels to int
        avg_image = avg_image.astype('uint8')
        # avg_image = cv2.cvtColor(avg_image,cv2.COLOR_BGR2RGB) # no need if cam format is RGB
        self.continue_stream()
        return avg_image
    
    def capture_single(self,grayscale=False, normalize=True, hue_mult=1, brightness_mult=1):
        """Captures single consecutive images, turns them into grayscale, normalize then computes and returns the average image."""
        w, h = self.capture_resolution
        
        self.stop_stream()
        self.capture_config()
        time.sleep(1)
        
        start_time = time.time()
        for i in range(5): # left 5 images for waiting to config
            frame = self.picam2.capture_buffer()
        time.sleep(self.shot_delay)
        end_time = time.time()
        capture_duration = round(end_time - start_time, 2)
        print(f'--- captured single image in {capture_duration} seconds ---')

        # reshape the frame and change type
        
        frame = np.frombuffer(frame, np.uint8).reshape(h, w,3)
        
        # hsv parameters
        if hue_mult != 1 or brightness_mult != 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            frame[...,1] = frame[...,1]*hue_mult
            frame[...,2] = frame[...,2]*brightness_mult # multiply by a factor of less than 1 to reduce the brightness
            frame =  cv2.cvtColor(frame, cv2.COLOR_HSV2BGR)
            
        # convert to grayscale
        if grayscale:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) 
            
        # normalize light, alpha and beta should be parametrized
        if normalize:
            frame = cv2.normalize(frame, None, alpha=30, beta=200, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_32F)

        # convert color channels to int
        frame = frame.astype('uint8') # must be here for opencv calculation
        
        # avg_image = cv2.cvtColor(avg_image,cv2.COLOR_BGR2RGB) # no need if cam format is RGB
        self.continue_stream()
        return frame

'''
64 cam data:
{'SensorTimestamp': 68424933423000, 'ScalerCrop': (2064, 2032, 5120, 2880), 'FocusFoM': 1590, 
'ColourCorrectionMatrix': (1.44386887550354, -0.2626706063747406, -0.18120574951171875, -0.37125107645988464, 1.456932544708252, -0.08567647635936737, -0.12658314406871796, -0.7002722024917603, 1.8268554210662842), 
'FrameDuration': 33324, 'AeLocked': False, 'AfPauseState': 0, 'AnalogueGain': 5.988304138183594, 
'ColourGains': (1.483496069908142, 2.294881582260132), 'ExposureTime': 32918, 'DigitalGain': 1.0, 
'AfState': 0, 'ColourTemperature': 3747, 'Lux': 286.1312255859375, 'SensorBlackLevels': (4096, 4096, 4096, 4096)}
'''
