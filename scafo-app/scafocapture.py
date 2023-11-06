import cv2
import numpy as np
import time
from datetime import datetime
from enum import Enum
from picamera.array import PiRGBArray
from picamera import PiCamera

class CapAvgMethod(Enum):
    AddWeighted = 1
    Max = 2
    Min = 3

class ScafoCapture:
    def __init__(self, capture_resolution=(4032, 3040), streaming_resolution=(1280,720), shot_delay=0.1, monitoring_delay=30): # i
        # highest 4608 x 2592
        self.capture_resolution = capture_resolution 
        self.streaming_resolution = streaming_resolution
        
        self.shot_delay = shot_delay
        self.monitoring_delay = monitoring_delay

        self.camera = cv2.VideoCapture(-1,cv2.CAP_V4L2) # 0+cv2.CAP_ANY
        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self.camera.set(cv2.CAP_PROP_EXPOSURE, 0) # Zero for automatic exposure

        self.streaming = False
        self.streaming_start_time = None
        self.monitoring = False

    def initialize_video_capture(self):
        self.camera = cv2.VideoCapture(0+cv2.CAP_ANY)
        self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0)
        self.camera.set(cv2.CAP_PROP_EXPOSURE, 0) # Zero for automatic exposure
        self.streaming_config()

    def streaming_config(self):
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.streaming_resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.streaming_resolution[1])

    def capture_config(self):
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, self.capture_resolution[0])
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, self.capture_resolution[1])

    def start_stream(self):
        self.streaming = True
        self.streaming_config()
        
        blank_img = np.zeros((self.streaming_resolution[1],self.streaming_resolution[0],3), np.uint8)
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
                read_success, frame = self.camera.read()
                if not read_success:
                    frame =  blank_img
                    break
                else:
                    _, buffer = cv2.imencode('.jpg', frame)
                    frame = buffer.tobytes()
            else:
               frame =  blank_img

            yield (b'--frame\r\n Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')  # concat frame one by one and show result

        
    
    def continue_stream(self):
        self.streaming_config()
        self.streaming = True

    def stop_stream(self):
        self.streaming = False
        # self.camera.release()
        # cv2.destroyAllWindows()

    def capture_n(self, n=5, grayscale=False, normalize=True, hue_mult=1, brightness_mult=1, avg_method=CapAvgMethod.AddWeighted):
        """Captures n consecutive images, turns them into grayscale, normalize then computes and returns the average image."""
        self.stop_stream()
        self.capture_config()
        image_data = []

        print(f'--- capturing {n} photos, averaging method: {avg_method} ---')

        start_time = time.time()
        for i in range(n):
            # capture image
            success, img = self.camera.read()
            if not success:
                print('error while capturing image')
            #print(f'image {i+1} captured')
            time.sleep(self.shot_delay)
            # append image to array
            image_data.append(img)
        
        end_time = time.time()
        capture_duration = round(end_time - start_time, 2)
        print(f'--- captured in {capture_duration} seconds ---')
        print(f'--- averaging now ---')
        
        # average the images
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
        
        self.continue_stream()
        return avg_image