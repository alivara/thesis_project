import time
import paho.mqtt.client as mqttClient
import json


class MQTTClient:
    def __init__(self, conf_file,MQMES_client_id, wait_time=5):
        # reading the config file
        self.server_address = conf_file['MQMES_server_address']
        self.server_port = conf_file['MQMES_server_port']
        self.username = conf_file['MQMES_username']
        self.password = conf_file['MQMES_password']
        self.client_id = MQMES_client_id
        self.wait_time = wait_time
        self.topics = [("scafo4/deformation/dev000", 0), ("scafo4/deformation/dev000/status", 0)]
        self.client = mqttClient.Client(self.client_id)
        self.client.username_pw_set(self.username, password=self.password)
        self.client.connected_flag = False
        self.client.on_connect = self.on_connect
        self.client.on_disconnect = self.on_disconnect
        self.dataMsg = {
                        "sensorID": conf_file['MQMES_client_id'],
                        "timestamp": int(time.time()),
                        "status": "1", #"0" (spento), "1" (ok), "2" (warning), "3" (alarm)
                        "comment": "... eventuali commenti ..."}

    def on_connect(self, client, userdata, flags, rc):
        if rc == 0:
            client.connected_flag = True

    def on_disconnect(self, client, userdata, flags, rc):
        client.connected_flag = False
        client.connect(self.server_address, port=self.server_port)
        client.loop_start()

    def connect(self):
        self.client.connect(self.server_address, port=self.server_port)
        self.client.loop_start()

        while not self.client.connected_flag:
            time.sleep(1)

    def run(self):
        dataMsg = json.dumps(self.dataMsg)
        try:
            print("Sending data to MQTT")
            statusMsg = json.dumps({"isConnected": 1})
            self.client.publish(self.topics[1][0], statusMsg)
            # sending dataMsg as message
            self.client.publish(self.topics[0][0], dataMsg)
            time.sleep(self.wait_time)
            print("Data has been sent to MQTT")

        except KeyboardInterrupt:
            print("Error in sending data to MQTT")
            self.client.disconnect()
            self.client.loop_stop()


