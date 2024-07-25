package com.sidra.nexus;

import org.ros.node.Node;
import org.ros.node.NodeMain;
import org.ros.node.NodeMainExecutor;
import org.ros.node.DefaultNode;
import org.ros.node.topic.Publisher;
import org.ros.node.topic.Subscriber;

import geometry_msgs.Twist;

public class RoboticsManager implements NodeMain {
    private Node node;
    private Publisher<Twist> velocityPublisher;
    private Subscriber<std_msgs.String> commandSubscriber;

    public RoboticsManager() {
        node = NodeMainExecutor.newNode("robotics_manager", this);
    }

    @Override
    public void onStart(Node node) {
        // Initialize robotics system
        velocityPublisher = node.newPublisher("cmd_vel", Twist._TYPE);
        commandSubscriber = node.newSubscriber("commands", std_msgs.String._TYPE);
    }

    @Override
    public void onShutdown(Node node) {
        // Shutdown robotics system
        velocityPublisher.shutdown();
        commandSubscriber.shutdown();
    }

    @Override
    public void onError(Node node, Throwable throwable) {
        // Handle error
        throwable.printStackTrace();
    }

    public void moveRobot(String direction) {
        // Move robot in specified direction
        Twist twist = velocityPublisher.newMessage();
        switch (direction) {
            case "forward":
                twist.getLinear().setX(1.0);
                break;
            case "backward":
                twist.getLinear().setX(-1.0);
                break;
            case "left":
                twist.getAngular().setZ(1.0);
                break;
            case "right":
                twist.getAngular().setZ(-1.0);
                break;
            default:
                twist.getLinear().setX(0.0);
                twist.getAngular().setZ(0.0);
                break;
        }
        velocityPublisher.publish(twist);
    }

    public void rotateRobot(double angle) {
        // Rotate robot by specified angle
        Twist twist = velocityPublisher.newMessage();
        twist.getAngular().setZ(angle);
        velocityPublisher.publish(twist);
    }

    public void stopRobot() {
        // Stop robot
        Twist twist = velocityPublisher.newMessage();
        twist.getLinear().setX(0.0);
        twist.getAngular().setZ(0.0);
        velocityPublisher.publish(twist);
    }

    public void startListeningForCommands() {
        commandSubscriber.addMessageListener(message -> {
            String command = message.getData();
            if (command.startsWith("move ")) {
                moveRobot(command.substring(5));
            } else if (command.startsWith("rotate ")) {
                rotateRobot(Double.parseDouble(command.substring(7)));
            } else if (command.equals("stop")) {
                stopRobot();
            }
        });
    }
}
