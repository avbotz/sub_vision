"""
Utilities for logging images.
"""
import datetime
import os
import logging
import rclpy.logging
import cv2

def log(image, prefix="", fast_log=False, file_ending="",):
    """
    Logs an image in images/.
    Will save based on current date and time.
    """
    if not file_ending:
        file_ending = "bmp" if fast_log else "png"
    save_location = f"images/{prefix}/{datetime.datetime.now().isoformat()}.{file_ending}"

    os.mkdir(f"images/{datetime.date.today()}")

    # Write images after making sure they exist
    if not image.data:
        rclpy.logging.get_logger("vision_logger").log(
            logging.ERROR, "Could not find logging image data."
        )

    if not cv2.imwrite(save_location, image):
        rclpy.logging.get_logger("vision_logger").log(
            logging.ERROR, "Could not write image to log directory."
        )
