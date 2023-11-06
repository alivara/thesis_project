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


## Important Note

The code contains several configuration options and functionality related to capturing, processing, and analyzing data from a camera system. Some parts of the code are specific to your project, and additional documentation or comments within the code may be required for a better understanding of the implementation.

Please make sure to adapt the code and configuration to your specific use case and project requirements. Feel free to update this README with more detailed instructions as needed.
