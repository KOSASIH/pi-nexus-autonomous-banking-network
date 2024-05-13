import numpy as np
import cv2

class DR:
    def __init__(self, model_path: str, device: str = 'cpu'):
        """
        Initialize the DR class with a pre-trained deep learning model.
        """
        self.model = cv2.dnn.readNetFromDarknet(model_path)
        self.device = device

    def detect(self, image: np.ndarray) -> List[Dict[str, Any]]:
        """
        Detect objects in an image using the pre-trained deep learning model.
        """
        blob = cv2.dnn.blobFromImage(image, 1 / 255.0, (416, 416), swapRB=True, crop=False)
        self.model.setInput(blob)
        outputs = self.model.forward(self.model.getUnconnectedOutLayersNames())

        boxes = []
        confidences = []
        class_ids = []

        for output in outputs:
            for detection in output:
                scores = detection[5:]
                class_id = np.argmax(scores)
                confidence = scores[class_id]

                if confidence > 0.5:
                    center_x = int(detection[0] * image.shape[1])
                    center_y = int(detection[1] * image.shape[0])
                    w = int(detection[2] * image.shape[1])
                    h = int(detection[3] * image.shape[0])

                    x = int(center_x - w / 2)
                    y = int(center_y - h / 2)

                    boxes.append([x, y, w, h])
                    confidences.append(float(confidence))
                    class_ids.append(class_id)

        indices = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)
        result = []

        for i in indices:
            box = boxes[i]
            x, y, w, h = box
            result.append({
                "class_id": class_ids[i],
                "confidence": confidences[i],
              "box": [x, y, w, h]
            })

        return result
