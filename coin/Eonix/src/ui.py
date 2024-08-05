# ui.py
import tkinter as tk
from eonix import Eonix

class EonixUI:
    def __init__(self, master):
        self.master = master
        self.eonix = Eonix()
        self.create_widgets()

    def create_widgets(self):
        self.balance_label = tk.Label(self.master, text="Balance: 0")
        self.balance_label.pack()

        self.send_button = tk.Button(self.master, text="Send", command=self.send_transaction)
        self.send_button.pack()

        self.mine_button = tk.Button(self.master, text="Mine", command=self.mine_block)
        self.mine_button.pack()

        self.recipient_entry = tk.Entry(self.master)
        self.recipient_entry.pack()

        self.amount_entry = tk.Entry(self.master)
        self.amount_entry.pack()

    def send_transaction(self):
        recipient = self.recipient_entry.get()
        amount = int(self.amount_entry.get())
        self.eonix.create_transaction(recipient, amount)
        self.update_balance()

    def mine_block(self):
        self.eonix.mine_block()
        self.update_balance()

    def update_balance(self):
        balance = self.eonix.get_balance()
        self.balance_label.config(text=f"Balance: {balance}")

root = tk.Tk()
eonix_ui = EonixUI(root)
root.mainloop()
