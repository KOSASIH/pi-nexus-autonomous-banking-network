import pyautogui

class Automation:
    def __init__(self, tasks):
        self.tasks = tasks

    def automate_tasks(self):
        # Automate repetitive tasks using RPA
        for task in self.tasks:
            pyautogui.click(task['x'], task['y'])
            pyautogui.typewrite(task['text'])
            pyautogui.press(task['key'])
