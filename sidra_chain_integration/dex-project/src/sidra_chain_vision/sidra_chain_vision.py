# sidra_chain_vision.py
import rospy
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
from tensorflow.keras.models import load_model

class SidraChainVision:
    def __init__(self):
        # Initialize ROS node
        rospy.init_node("sidra_chain_vision")

        # Create ROS subscribers and publishers
        self.image_sub = rospy.Subscriber("image", Image, self.image_callback)
        self.object_pub = rospy.Publisher("object", String, queue_size=10)
        self.bbox_pub = rospy.Publisher("bbox", String, queue_size=10)

        # Create CvBridge object
        self.bridge = CvBridge()

        # Load object detection model
        self.model = load_model("yolov3.h5")

    def image_callback(self, msg):
        # Process image message
        cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        objects, bboxes = self.detect_objects(cv_image)
        self.object_pub.publish(objects)
        self.bbox_pub.publish(bboxes)

    def detect_objects(self, image):
        # Detect objects using YOLOv3
        input_image = cv2.resize(image, (416, 416))
        input_image = input_image / 255.0
        input_image = np.expand_dims(input_image, axis=0)

        outputs = self.model.predict(input_image)
        outputs = np.squeeze(outputs)

        objects = []
        bboxes = []
        for output in outputs:
            scores = output[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5 and class_id == 0:  # person class
                x, y, w, h = output[:4]
                x_min = int(x - w / 2)
                y_min = int(y - h / 2)
                x_max = int(x + w / 2)
                y_max = int(y + h / 2)
                objects.append("person")
                bboxes.append(f"{x_min},{y_min},{x_max},{y_max}")

        return objects, bboxes

if __name__ == "__main__":
    sidra_chain_vision = SidraChainVision()
    rospy.spin()
