#include <opencv2/opencv.hpp>
#include <iostream>

int main() {
    cv::Mat image = cv::imread("image.jpg");
    cv::Mat gray;
    cv::cvtColor(image, gray, cv::COLOR_BGR2GRAY);

    cv:: CascadeClassifier faceDetector("haarcascade_frontalface_default.xml");
    std::vector<cv::Rect> faces;
    faceDetector.detectMultiScale(gray, faces);

    for (size_t i = 0; i < faces.size(); i++) {
        cv::rectangle(image, faces[i], cv::Scalar(0, 255, 0), 2);
    }

    cv::imshow("Faces", image);
    cv::waitKey(0);
    cv::destroyAllWindows();
    return 0;
}
