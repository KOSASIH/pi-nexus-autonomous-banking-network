// Robotics.java
import java.util.ArrayList;
import java.util.List;

public class Robotics {
    public static void main(String[] args) {
        List<String> instructions = new ArrayList<>();
        instructions.add("move forward 10");
        instructions.add("turn left 90");
        instructions.add("move backward 5");
        Robot robot = new Robot();
        robot.executeInstructions(instructions);
    }
}

class Robot {
    public void executeInstructions(List<String> instructions) {
        for (String instruction : instructions) {
            String[] parts = instruction.split(" ");
            if (parts[0].equals("move")) {
                if (parts[1].equals("forward")) {
                    moveForward(Integer.parseInt(parts[2]));
                } else if (parts[1].equals("backward")) {
                    moveBackward(Integer.parseInt(parts[2]));
                }
            } else if (parts[0].equals("turn")) {
                if (parts[1].equals("left")) {
                    turnLeft(Integer.parseInt(parts[2]));
                } else if (parts[1].equals("right")) {
                    turnRight(Integer.parseInt(parts[2]));
                }
            }
        }
    }

    private void moveForward(int distance) {
        System.out.println("Moving forward " + distance + " units");
    }

    private void moveBackward(int distance) {
        System.out.println("Moving backward " + distance + " units");
    }

    private void turnLeft(int angle) {
        System.out.println("Turning left " + angle + " degrees");
    }

    private void turnRight(int angle) {
        System.out.println("Turning right " + angle + " degrees");
    }
}
