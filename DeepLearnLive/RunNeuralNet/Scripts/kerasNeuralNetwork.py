import os
import time
import cv2
import sys
import numpy
import random
import argparse
import logging
import pyigtl

FLAGS = None

def main():
    print(FLAGS.model_directory)
    try:
        networkModuleName = FLAGS.model_name
        sys.path.append(os.path.join(FLAGS.model_directory,os.pardir))
        importStatement = "from " + networkModuleName + " import " + networkModuleName + " as NeuralNetwork"
        exec(importStatement,globals())
    except ModuleNotFoundError:
        logging.info("Could not find model folder " + str(FLAGS.model_name))
        errorMessage = "Could not find model folder " + str(FLAGS.model_name)
        print(errorMessage)
    except:
        print(FLAGS.model_name)
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
    model_name = FLAGS.model_name
    modelFolder =FLAGS.model_directory
    currentTime = time.time()
    model = NeuralNetwork()
    model.loadModel(modelFolder,model_name)

    print("Server starting...")
    if FLAGS.outgoing_host == "localhost":
        server = pyigtl.OpenIGTLinkServer(port=FLAGS.outgoing_port,local_server=True)
    else:
        server = pyigtl.OpenIGTLinkServer(port=FLAGS.outgoing_port, local_server=False)

    #server = pyigtl.OpenIGTLinkServer(port=FLAGS.outgoing_port)
    server.start()
    print("Server running on " + str(server.host) + " : " + str(server.port) + "...")


    print("Client starting...")
    #client = pyigtl.OpenIGTLinkClient(host=FLAGS.incoming_host, port=FLAGS.incoming_port)
    client = pyigtl.OpenIGTLinkClient(host="localhost", port=FLAGS.incoming_port)
    client.start()
    print(FLAGS.incoming_host)
    print(FLAGS.incoming_port)
    print("Client running...")
    lastMessageTime = time.time()
    ImageReceived = False
    frameCount = 0
    try:
       while (not ImageReceived) or (time.time() - lastMessageTime < FLAGS.timeout):
        #while (not ImageReceived) or (time.time() - lastMessageTime < FLAGS.timeout):

            #if server.is_connected() and client.is_connected():

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
                        print(networkOutput)
                        print(time.time())
                        if FLAGS.output_type == 'STRING':
                            #print('got here')
                            labelMessage = pyigtl.StringMessage(networkOutput, device_name=FLAGS.device_name)
                            server.send_message(labelMessage)
                            #print('sent message')
                        elif FLAGS.output_type == 'IMAGE':
                            labelMessage = pyigtl.ImageMessage(networkOutput, device_name=FLAGS.device_name)
                            server.send_message(labelMessage)
                        elif FLAGS.output_type == 'TRANSFORM':
                            pass

                        print(frameCount)
                    if message._message_type == "STRING":
                        print("Received stop message")
                        text = message.string
                        if text == "STOP":
                            client.stop()
                            server.stop()
            else:
                pass
            time.sleep(0.25)
       client.stop()
       server.stop()

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
      default='STRING',
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
      default=18946,
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
      default='Text',
      help='The name of the node that the network output should be sent to'
  )
FLAGS, unparsed = parser.parse_known_args()
main()