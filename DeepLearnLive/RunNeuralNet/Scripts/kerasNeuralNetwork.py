import os
import time
import cv2
import sys
import numpy
import random
import argparse
import logging
import SimpleITK as sitk
import tensorflow
import tensorflow.keras
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras import layers
from tensorflow.keras.models import model_from_json
#from pyIGTLink import pyIGTLink
import pyigtl

FLAGS = None

def main():
    try:
        networkModuleName = FLAGS.network_module_name
        sys.path.append("C:/Users/hisey/Documents/DeepLearnLive/Networks/CNN_LSTM/")
        importStatement = "from " + networkModuleName + " import " + networkModuleName + " as NeuralNetwork"
        exec(importStatement,globals())
        print('got here')
    except ModuleNotFoundError:
        logging.info("Could not find model folder " + str(FLAGS.model_name))
        errorMessage = "Could not find model folder " + str(FLAGS.model_name)
        print(errorMessage)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    model_name = FLAGS.model_name
    modelFolder =FLAGS.model_directory
    currentTime = time.time()
    model = NeuralNetwork()
    model.loadModel(modelFolder,model_name)

    if FLAGS.outgoing_host == "localhost":
        print("Server starting...")
        server = pyigtl.OpenIGTLinkServer(port=FLAGS.outgoing_port)
        server.start()
        print("Server running...")


    print("Client starting...")
    client = pyigtl.OpenIGTLinkClient(host=FLAGS.incoming_host, port=FLAGS.incoming_port)
    client.start()
    print("Client running...")
    lastMessageTime = time.time()
    ImageReceived = False
    frameCount = 0
    try:
        while (not ImageReceived) or (ImageReceived and time.time() - lastMessageTime < FLAGS.timeout):
            messages = client.get_latest_messages()
            if len(messages) > 0:
                for message in messages:
                    if message._message_type == "IMAGE":
                        frameCount +=1
                        ImageReceived = True
                        lastMessageTime = time.time()
                        image = message.image
                        image = image[0]
                        print(time.time())
                        (networkOutput) = model.predict(image)
                        print(time.time())
                        if FLAGS.output_type == 'STRING':
                            labelMessage = pyigtl.StringMessage(networkOutput, device_name=FLAGS.device_name)
                            server.send_message(labelMessage)
                        elif FLAGS.output_type == 'IMAGE':
                            labelMessage = pyigtl.ImageMessage(networkOutput, device_name=FLAGS.device_name)
                            server.send_message(labelMessage)
                        elif FLAGS.output_type == 'TRANSFORM':
                            pass

                        print(frameCount)
            time.sleep(0.25)
    except KeyboardInterrupt:
        pass

if __name__ == '__main__':
  parser = argparse.ArgumentParser()
  parser.add_argument(
      '--network_module_name',
      type=str,
      default='',
      help='Name of module that defines the model and the predict function.'
  )
  parser.add_argument(
      '--model_name',
      type=str,
      default='',
      help='Name of model.'
  )
  parser.add_argument(
      '--model_directory',
      type=str,
      default='',
      help='Location of model.'
  )
  parser.add_argument(
      '--output_type',
      type=str,
      default='IMAGE',
      help='type of output your model generates'
  )
  parser.add_argument(
      '--timeout',
      type=int,
      default=15,
      help='Number of seconds before network stops waiting for new image'
  )
  parser.add_argument(
      '--incoming_host',
      type=str,
      default='localhost',
      help='Name of model.'
  )
  parser.add_argument(
      '--incoming_port',
      type=int,
      default=18945,
      help='Location of model.'
  )
  parser.add_argument(
      '--outgoing_host',
      type=str,
      default='localhost',
      help='type of output your model generates'
  )
  parser.add_argument(
      '--outgoing_port',
      type=int,
      default=18944,
      help='Number of seconds before network stops waiting for new image'
  )
  parser.add_argument(
      '--device_name',
      type=str,
      default='LabelNode',
      help='The name of the node that the network output should be sent to'
  )
FLAGS, unparsed = parser.parse_known_args()
main()