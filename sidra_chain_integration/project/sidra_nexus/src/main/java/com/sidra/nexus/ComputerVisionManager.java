package com.sidra.nexus;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class ComputerVisionManager {
    public void processImage(String filePath) {
        // Load image
        Mat image = Imgcodecs.imread(filePath);

        // Convert image to grayscale
        Mat grayImage = new Mat();
        Imgproc.cvtColor(image, grayImage, Imgproc.COLOR_BGR2GRAY);

        // Apply thresholding
        Mat thresholdImage = new Mat();
        Imgproc.threshold(grayImage, thresholdImage, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);

        // Find contours
        Mat contoursImage = new Mat();
        Imgproc.findContours(thresholdImage, contoursImage, new Mat(), Imgproc.RETR_EXTERNAL, Imgproc.CHAIN_APPROX_SIMPLE);

        // Draw contours
        Imgproc.drawContours(image, contoursImage, -1, new Scalar(0, 255, 0), 2);

        // Save output
        Imgcodecs.imwrite("output.jpg", image);
    }

    public void detectFaces(String filePath) {
        // Load image
        Mat image = Imgcodecs.imread(filePath);

        // Load face detection cascade
        String cascadeFilePath = "haarcascade_frontalface_default.xml";
        Mat faceCascade = Imgcodecs.imread(cascadeFilePath);

        // Detect faces
        Mat facesImage = new Mat();
        Imgproc.HaarDetectObjects(image, faceCascade, facesImage, new Scalar(0, 255, 0), 1.1, 2, 0, new Point(0, 0));

        // Draw rectangles around faces
        for (int i = 0; i < facesImage.rows(); i++) {
            double v = facesImage.get(i, 0)[0];
            Point p = new Point(v, facesImage.get(i, 1)[0]);
            Imgproc.rectangle(image, p, new Point(p.x + facesImage.get(i, 2)[0], p.y + facesImage.get(i, 3)[0]), new Scalar(0, 255, 0), 2);
        }

        // Save output
        Imgcodecs.imwrite("output.jpg", image);
    }
}
