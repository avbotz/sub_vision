from sub_vision_interfaces.srv import Vision
from sub_vision_interfaces.msg import Task

import rclpy
import rclpy.qos as qos
import sensor_msgs.msg

import sub_vision_yolov5.image_logger as image_logger

bridge = None # see main

class YOLOV5VisionService(rclpy.node.Node):
    def __init__(self):
        # Init with name
        super().__init__("sub_vision_yolov5")

        # Init services
        self.vision_service = self.create_service(
            Vision,
            "vision",
            self.detect_callback
        )

        # Declare and get parameters
        self.declare_parameter("SIM", False)
        self.sim = self.get_parameter("SIM")
        print("SIM:", self.sim)

        # Setup receiving camera images.
        # TODO: Just publish the actual images to the same topic, then
        # this ternary won't be necessary
        self._front_camera_topic = "/nemo/front_camera/image" if self.sim else "front_camera"
        self._down_camera_topic = "/nemo/down_camera/image" if self.sim else "down_camera"

        self.front_sub = self.create_subscription(
            sensor_msgs.msg.Image,
            self._front_camera_topic,
            self.handle_front_capture,
            qos.QoSProfile()
        )

        self.down_sub = self.create_subscription(
            sensor_msgs.msg.Image,
            self._down_camera_topic,
            self.handle_down_capture,
            qos.QoSProfile()
        )

    def handle_front_capture(self, image: sensor_msgs.msg.Image):
        print("front")
        cv_image = bridge.imgmsg_to_cv2(image)
        image_logger.log(cv_image)

    def handle_down_capture(self, image: sensor_msgs.msg.Image):
        print("down")
        cv_image = bridge.imgmsg_to_cv2(image)
        image_logger.log(cv_image)


def main(args=None):
    rclpy.init(args=args)

    from cv_bridge import CvBridge
    global bridge
    bridge = CvBridge()

    service = YOLOV5VisionService()
    rclpy.spin(service)
    rclpy.shutdown()

if __name__ == "__main__":
    main()
