"""
============================
Image sending server
============================

Simple application that starts a server that provides an image stream with a bright circle moving

"""

import pyigtl  # pylint: disable=import-error

from time import sleep
import numpy as np
from math import sin

server = pyigtl.OpenIGTLinkServer(port=18944)

image_size = [500, 300]
radius = 60

timestep = 0
while True:

    if not server.is_connected():
        # Wait for client to connect
        print("Wait for client to connect...")
        sleep(0.1)
        continue

    timestep += 1

    # Send message
    message = f"time: {timestep}"
    string_message = pyigtl.StringMessage(message, device_name="string_message")
    server.send_message(message=string_message, wait=True)

    #response = server.wait_for_message(device_name="modified_message", timeout=-1)
    #print(response)
    # Since we wait until the message is actually sent, the message queue will not be flooded
