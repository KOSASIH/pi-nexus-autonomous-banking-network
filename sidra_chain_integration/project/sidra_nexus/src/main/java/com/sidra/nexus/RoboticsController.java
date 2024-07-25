package com.sidra.nexus;

import java.awt.AWTException;
import java.awt.Robot;
import java.awt.event.InputEvent;

public class RoboticsController {
    private Robot robot;

    public RoboticsController() throws AWTException {
        robot = new Robot();
    }

    public void moveMouse(int x, int y) {
        robot.mouseMove(x, y);
    }

    public void clickMouse() {
        robot.mousePress(InputEvent.BUTTON1_MASK);
        robot.mouseRelease(InputEvent.BUTTON1_MASK);
    }

    public void typeText(String text) {
        for (char c : text.toCharArray()) {
            int keyCode = KeyEvent.getExtendedKeyCodeForChar(c);
            robot.keyPress(keyCode);
            robot.keyRelease(keyCode);
        }
    }
}
