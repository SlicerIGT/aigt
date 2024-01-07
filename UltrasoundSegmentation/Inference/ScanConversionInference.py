"""
Implements an OpenIGTLink client that receives ultrasound (pyigtl.ImageMessage) and sends prediction/segmentation (pyigtl.ImageMessage).
Transform messages (pyigtl.TransformMessage) are also received and sent to the server, but the device name is changed by replacing Image to Prediction.
This is done to ensure that the prediction is visualized in the same position as the ultrasound image.

Arguments:
    model: Path to the torchscript file you intend to use for segmentation. The model must be a torchscript model that takes a single image as input and returns a single image as output.
    input device name: This is the device name the client is listening to
    output device name: The device name the client outputs to
    host: Server's IP the client connects to.
    input port: Port used for receiving data from the PLUS server over OpenIGTLink
    output port: Port used for sending data to Slicer over OpenIGTLink
"""

import argparse
import cv2
import json
import logging
import numpy as np
import traceback
import sys
import pyigtl
import torch
import yaml

from pathlib import Path
from scipy.ndimage import map_coordinates
from scipy.spatial import Delaunay

ROOT = Path(__file__).parent.resolve()

# Parse command line arguments
def ScanConversionInference():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, help="Path to torchscript model file.")
    parser.add_argument("--scanconversion_config", type=str, help="Path to scan conversion config (.yaml) file. Optional.")
    parser.add_argument("--input-device-name", type=str, default="Image_Image")
    parser.add_argument("--output-device-name", type=str, default="Prediction")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--input-port", type=int, default=18944)
    parser.add_argument("--output-port", type=int, default=18945)
    parser.add_argument("--log_file", type=str, default=None, help="Path to log file. Optional.")
    try:
        args = parser.parse_args()
    except SystemExit as err:
        traceback.print_exc()
        sys.exit(err.code)

    if args.log_file:
        logging.basicConfig(filename=args.log_file, filemode='w', level=logging.INFO)
    else:
        logging.basicConfig(level=logging.INFO)
    
    run_client(args)

def run_client(args):
    """
    Runs the client in an infinite loop, waiting for messages from the server. Once a message is received,
    the message is processed and the inference is sent back to the server as a pyigtl ImageMessage.
    """
    input_client = pyigtl.OpenIGTLinkClient(host=args.host, port=args.input_port)
    output_server = pyigtl.OpenIGTLinkServer(port=args.output_port)
    model = None

    # Load pytorch model

    logging.info("Loading model...")
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_path = args.model if Path(args.model).is_absolute() else f'{str(ROOT)}/{args.model}'
    extra_files = {"config.json": ""}
    model = torch.jit.load(model_path, _extra_files=extra_files).to(device)
    config = json.loads(extra_files["config.json"])
    input_size = config["shape"][-1]
    logging.info("Model loaded")

    # If scan conversion is enabled, compute x_cart, y_cart, vertices, and weights for conversion and interpolation

    if args.scanconversion_config:
        logging.info("Loading scan conversion config...")
        with open(args.scanconversion_config, "r") as f:
            scanconversion_config = yaml.safe_load(f)
        x_cart, y_cart = scan_conversion_inverse(scanconversion_config)
        logging.info("Scan conversion config loaded")
    else:
        scanconversion_config = None
        x_cart = None
        y_cart = None
        logging.info("Scan conversion config not found")

    if x_cart is not None and y_cart is not None:
        vertices, weights = scan_interpolation_weights(scanconversion_config)
        mask_array = curvilinear_mask(scanconversion_config)
    else:
        vertices = None
        weights = None
        mask_array = None

    while True:
        messages = input_client.get_latest_messages()
        for message in messages:
            if message.device_name == args.input_device_name:  # Image message
                if model is None:
                    logging.error("Model not loaded. Exiting...")
                    break
                
                # Resize image to model input size
                orig_img_size = message.image.shape
                image = preprocess_input(message.image, input_size, scanconversion_config, x_cart, y_cart).to(device)
            
                # Run inference
                with torch.inference_mode():
                    prediction = model(image)

                if isinstance(prediction, list):
                    prediction = prediction[0]
                    
                prediction = torch.nn.functional.softmax(prediction, dim=1)
                prediction = postprocess_prediction(prediction, orig_img_size, scanconversion_config, vertices, weights, mask_array)

                image_message = pyigtl.ImageMessage(prediction, device_name=args.output_device_name)
                output_server.send_message(image_message, wait=True)

            if message.message_type == "TRANSFORM" and "Image" in message.device_name:  # Image transform message
                output_tfm_name = message.device_name.replace("Image", "Prediction")
                tfm_message = pyigtl.TransformMessage(message.matrix, device_name=output_tfm_name)
                output_server.send_message(tfm_message, wait=True)

def preprocess_input(image, input_size, scanconversion_config=None, x_cart=None, y_cart=None):
    if scanconversion_config is not None:
        # Scan convert image from curvilinear to linear
        num_samples = scanconversion_config["num_samples_along_lines"]
        num_lines = scanconversion_config["num_lines"]
        converted_image = np.zeros((1, num_lines, num_samples))
        converted_image[0, :, :] = map_coordinates(image[0, :, :], [x_cart, y_cart], order=1, mode='constant', cval=0.0)
        # Squeeze converted image to remove channel dimension
        converted_image = converted_image.squeeze()
    else:
        converted_image = cv2.resize(image[0, :, :], (input_size, input_size)) / 255  # default is bilinear
    
    converted_image = torch.from_numpy(converted_image).unsqueeze(0).unsqueeze(0).float()
    return converted_image

def postprocess_prediction(prediction, original_size, scanconversion_config=None, vertices=None, weights=None, mask_array=None):
    if scanconversion_config is not None:
        # Scan convert prediction from linear to curvilinear
        prediction = prediction.squeeze().detach().cpu().numpy() * 255
        # Make sure prediction data type is uint8
        # prediction = prediction.astype(np.uint8)[np.newaxis, ...]
        prediction = scan_convert(prediction[1], scanconversion_config, vertices, weights)
        if mask_array is not None:
            prediction = prediction * mask_array
        prediction = prediction.astype(np.uint8)[np.newaxis, ...]
    else:
        prediction = prediction.squeeze().detach().cpu().numpy() * 255
        prediction = cv2.resize(prediction[1], (original_size[2], original_size[1]))
        prediction = prediction.astype(np.uint8)[np.newaxis, ...]
    return prediction

def scan_conversion_inverse(scanconversion_config):
    """
    Compute cartesian coordianates for inverse scan conversion.
    Mapping from curvilinear image to a rectancular image of scan lines as columns.
    The returned cartesian coordinates can be used to map the curvilinear image to a rectangular image using scipy.ndimage.map_coordinates.

    Args:
        scanconversion_config (dict): Dictionary with scan conversion parameters.

    Rerturns:
        x_cart (np.ndarray): x coordinates of the cartesian grid.
        y_cart (np.ndarray): y coordinates of the cartesian grid.

    Example:
        >>> x_cart, y_cart = scan_conversion_inverse(scanconversion_config)
        >>> scan_converted_image = map_coordinates(ultrasound_data[0, :, :, 0], [x_cart, y_cart], order=3, mode="nearest")
        >>> scan_converted_segmentation = map_coordinates(segmentation_data[0, :, :, 0], [x_cart, y_cart], order=0, mode="nearest")
    """

    # Create sampling points in polar coordinates

    initial_radius = np.deg2rad(scanconversion_config["angle_min_degrees"])
    final_radius = np.deg2rad(scanconversion_config["angle_max_degrees"])
    radius_start_px = scanconversion_config["radius_start_pixels"]
    radius_end_px = scanconversion_config["radius_end_pixels"]

    theta, r = np.meshgrid(np.linspace(initial_radius, final_radius, scanconversion_config["num_samples_along_lines"]),
                           np.linspace(radius_start_px, radius_end_px, scanconversion_config["num_lines"]))

    # Convert the polar coordinates to cartesian coordinates

    x_cart = r * np.cos(theta) + scanconversion_config["center_coordinate_pixel"][0]
    y_cart = r * np.sin(theta) + scanconversion_config["center_coordinate_pixel"][1]

    return x_cart, y_cart

def scan_interpolation_weights(scanconversion_config):
    image_size = scanconversion_config["curvilinear_image_size"]

    x_cart, y_cart = scan_conversion_inverse(scanconversion_config)
    triangulation = Delaunay(np.vstack((x_cart.flatten(), y_cart.flatten())).T)

    grid_x, grid_y = np.mgrid[0:image_size, 0:image_size]
    simplices = triangulation.find_simplex(np.vstack((grid_x.flatten(), grid_y.flatten())).T)
    vertices = triangulation.simplices[simplices]

    X = triangulation.transform[simplices, :2]
    Y = np.vstack((grid_x.flatten(), grid_y.flatten())).T - triangulation.transform[simplices, 2]
    b = np.einsum('ijk,ik->ij', X, Y)
    weights = np.c_[b, 1 - b.sum(axis=1)]

    return vertices, weights

def scan_convert(linear_data, scanconversion_config, vertices, weights):
    """
    Scan convert a linear image to a curvilinear image.

    Args:
        linear_data (np.ndarray): Linear image to be scan converted.
        scanconversion_config (dict): Dictionary with scan conversion parameters.

    Returns:
        scan_converted_image (np.ndarray): Scan converted image.
    """
    
    z = linear_data.flatten()
    zi = np.einsum('ij,ij->i', np.take(z, vertices), weights)

    image_size = scanconversion_config["curvilinear_image_size"]
    return zi.reshape(image_size, image_size)

def curvilinear_mask(scanconversion_config):
    """
    Generate a binary mask for the curvilinear image with ones inside the scan lines area and zeros outside.

    Args:
        scanconversion_config (dict): Dictionary with scan conversion parameters.

    Returns:
        mask_array (np.ndarray): Binary mask for the curvilinear image with ones inside the scan lines area and zeros outside.
    """
    angle1 = 90.0 + (scanconversion_config["angle_min_degrees"])
    angle2 = 90.0 + (scanconversion_config["angle_max_degrees"])
    center_rows_px = scanconversion_config["center_coordinate_pixel"][0]
    center_cols_px = scanconversion_config["center_coordinate_pixel"][1]
    radius1 = scanconversion_config["radius_start_pixels"]
    radius2 = scanconversion_config["radius_end_pixels"]
    image_size = scanconversion_config["curvilinear_image_size"]

    mask_array = np.zeros((image_size, image_size), dtype=np.int8)
    mask_array = cv2.ellipse(mask_array, (center_cols_px, center_rows_px), (radius2, radius2), 0.0, angle1, angle2, 1, -1)
    mask_array = cv2.circle(mask_array, (center_cols_px, center_rows_px), radius1, 0, -1)
    # Convert mask_array to uint8
    mask_array = mask_array.astype(np.uint8)

    # Erode mask by one pixel to avoid interpolation artifacts at the edges
    mask_array = cv2.erode(mask_array, np.ones((3, 3), np.uint8), iterations=1)
    
    return mask_array

if __name__ == "__main__":
    ScanConversionInference()
