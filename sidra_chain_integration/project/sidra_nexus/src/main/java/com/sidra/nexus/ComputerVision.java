package com.sidra.nexus;

import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class ComputerVision {
    static {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
    }

    public void detectFaces(String imagePath) {
        Mat image = Imgcodecs.imread(imagePath);
        Mat gray = new Mat();
        Imgproc.cvtColor(image, gray, Imgproc.COLOR_BGR2GRAY);
        Imgproc.equalizeHist(gray, gray);
        // Face detection using Haar cascades
        String faceCascadePath = "haarcascade_frontalface_default.xml";
        Mat faceCascade = Imgcodecs.imread(faceCascadePath);
        MatOfRect faceDetections = new MatOfRect();
        faceCascade.detectMultiScale(gray, faceDetections);
        for (Rect rect : faceDetections.toArray()) {
            Imgproc.rectangle(image, new Point(rect.x, rect.y), new Point(rect.x + rect.width, rect.y + rect.height), new Scalar(0, 255, 0));
        }
        Imgcodecs.imwrite("output.jpg", image);
    }
}
