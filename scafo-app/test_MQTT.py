
import yaml
import os
import json

# with open('/home/pi/scafo4.0/flask/scafo-app/static/output/MQTTmsg15-53-49.json') as json_data:
#     d = json.load(json_data)
#     json_data.close()
    

PATH = os.path.dirname(__file__)
with open(PATH+'/config.yaml',) as file:
    conf_file = yaml.load(file, Loader=yaml.FullLoader)

from mqtt_cl import MQTTClient
MQMES = MQTTClient(conf_file = conf_file['MQTT'])
MQMES.connect() # on after debug

with open('/home/pi/scafo4.0/flask/scafo-app/static/output/MQTTmsg15-53-49.json') as json_data:
    MQMES.dataMsg = json.load(json_data)
    json_data.close()

while True:
# try:
    # MQMES.dataMsg = convert_numpy_to_list(MQMES.dataMsg)
    MQMES.run()
# except:
    # print('Erro')
        

