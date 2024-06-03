import tkinter as tk
from tkinter import filedialog, messagebox, ttk

from ui_designer import UIDesigner
from wallet_notification_system import NotificationSystem
from wallet_personalization_engine import PersonalizationEngine


class WalletApp:
    def __init__(self, root):
        self.root = root

        # Initialize UI designer
        self.ui_designer = UIDesigner(root)

        # Initialize personalization engine
        self.user_data = pd.read_csv("user_data.csv")
        self.personalization_engine = PersonalizationEngine(self.user_data)
        self.personalization_engine.train_model()

        # Initialize notification system
        self.smtp_server = "smtp.gmail.com"
        self.smtp_port = 465
        self.sender_email = "your_email@gmail.com"
        self.sender_password = "your_password"
        self.notification_system = NotificationSystem(
            self.smtp_server, self.smtp_port, self.sender_email, self.sender_password
        )

        # Set up user interaction methods
        self.ui_designer.settings_option.bind("<<ComboboxSelected>>", self.change_theme)
        self.ui_designer.upload_button.configure(command=self.upload_transaction_data)
        self.ui_designer.transaction_treeview.bind(
            "<<TreeviewSelect>>", self.view_transaction_details
        )

    def change_theme(self, event):
        # Change the theme of the wallet app
        theme = self.ui_designer.settings_option.get()

        if theme == "Light":
            self.ui_designer.set_theme("light")
        elif theme == "Dark":
            self.ui_designer.set_theme("dark")

    def upload_transaction_data(self):
        # Upload transaction data to the wallet app
        file_path = filedialog.askopenfilename(filetypes=[("CSV files", "*.csv")])

        if file_path:
            self.transaction_data = pd.read_csv(file_path)
            self.ui_designer.display_transaction_data(self.transaction_data)

    def view_transaction_details(self, event):
        # View details of the selected transaction
        transaction_index = self.ui_designer.transaction_treeview.selection()[0]
        transaction_details = self.transaction_data.iloc[transaction_index]

        messagebox.showinfo("Transaction Details", transaction_details.to_string())


if __name__ == "__main__":
    root = tk.Tk()
    wallet_app = WalletApp(root)
    root.mainloop()
