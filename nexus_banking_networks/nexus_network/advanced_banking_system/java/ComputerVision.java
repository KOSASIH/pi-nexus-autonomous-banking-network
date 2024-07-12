// ComputerVision.java
import org.opencv.core.Core;
import org.opencv.core.Mat;
import org.opencv.core.Point;
import org.opencv.core.Scalar;
import org.opencv.imgcodecs.Imgcodecs;
import org.opencv.imgproc.Imgproc;

public class ComputerVision {
    public static void main(String[] args) {
        System.loadLibrary(Core.NATIVE_LIBRARY_NAME);
        Mat image = Imgcodecs.imread("image.jpg");
        Imgproc.cvtColor(image, image, Imgproc.COLOR_BGR2GRAY);
        Imgproc.threshold(image, image, 0, 255, Imgproc.THRESH_BINARY_INV + Imgproc.THRESH_OTSU);
        Imgcodecs.imwrite("output.jpg", image);
    }
}
