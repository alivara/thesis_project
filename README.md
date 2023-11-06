# Scafo4.0

This repository contains the source code for an application that is used for monitoring and analyzing data from a camera system. Below are the instructions and details for setting up and using this application.

## Prerequisites
- Python 3
- Various Python libraries (specified in `requirements.txt`)
- MQTT server
- Raspberry Pi with a camera (or equivalent camera system)
- Motion sensor (optional)
- USB device (optional)

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/yourusername/yourrepository.git
   ```

2. Install the required Python libraries using pip:

   ```bash
   pip install -r requirements.txt
   ```

## Configuration

1. Make sure to have the MQTT server set up and its configuration in the `config.yaml` file.

2. Configure other settings in the `config.yaml` file as needed.

## Usage

1. Start the Flask application by running the following command:

   ```bash
   python app.py
   ```

2. Access the web application through your web browser. The main page should be available at `http://localhost:yourport/`. 

3. You can perform the following tasks using the web interface:

   - Setup: Configure camera settings, project patterns, and capture the baseline image.
   - Monitoring: Start and stop the monitoring process to capture and analyze data from the camera.
   - Deformation Detection: Evaluate noise, set noise threshold, and visualize noise evaluation results.
   - Debug: This section is for debugging and testing features of the application.

### Setup Process:

The setup process is responsible for configuring and preparing the application for monitoring. This includes configuring the camera, capturing a baseline image, and setting up other parameters.

1. **Thread `setup`**: This thread handles the setup process. It listens for a setup request from the web interface, which includes configuring camera settings and capturing a baseline image. The setup process is initiated by the user through the web interface.

2. **Capturing a Baseline Image**: This is done to establish a reference point for later comparisons during the monitoring process. The captured image is stored for further use in the monitoring process.

### Monitoring:

The monitoring process is the core of your application. It continuously captures images from the camera, analyzes the data, and sends notifications based on the analysis results.

1. **Thread `start_monitoring`**: This thread starts the monitoring process. It initiates the continuous capture of images and analysis of data. It also handles the interaction with the Motion Sensor, if available.

2. **Motion Sensor (Optional)**: If a motion sensor is available, it can send data related to motion events. The motion sensor's data is collected and processed concurrently with the camera images to detect any motion-related events.

3. **Capturing and Processing Images**: The monitoring thread captures images from the camera system and processes them for analysis. The analysis includes detecting LED tags and checking for deformation in the captured images.

4. **Deformation Analysis**: Deformation analysis involves checking for changes in the captured images compared to the baseline image. Deformation could indicate physical changes or shifts in the setup.

5. **Notification via MQTT**: When significant events, such as deformation or motion, are detected, the application sends notifications via MQTT (Message Queuing Telemetry Transport) to the MQTT server. These notifications include information about the detected events and relevant data.

6. **Data Storage**: Some data, such as the captured images and analysis results, are stored in appropriate directories or transferred to a USB device, depending on the application's configuration.

7. **Continual Monitoring**: The monitoring process runs continuously, capturing images at defined intervals and analyzing the data. It stops when the user initiates the "stop monitoring" command through the web interface.

### Threads and Concurrency:

Threads are used to ensure that different parts of the application can run concurrently. For example, capturing images and analyzing deformation can happen simultaneously. This concurrency improves the application's responsiveness and overall performance.

It's important to manage shared resources, such as data and image files, to prevent conflicts and data corruption. Your code handles this by using threads for different tasks and appropriately managing data transfer between threads.

Overall, threads are a fundamental part of your application, enabling it to efficiently capture, analyze, and notify users about events detected by the camera system.
## Important Note

The code contains several configuration options and functionality related to capturing, processing, and analyzing data from a camera system. Some parts of the code are specific to your project, and additional documentation or comments within the code may be required for a better understanding of the implementation.

