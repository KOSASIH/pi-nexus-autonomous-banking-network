class Device:
    def __init__(self, name, topic):
        self.name = name
        self.topic = topic

    def read_sensor(self):
        # Kode untuk membaca data sensor
        return 123

    def send_command(self, command):
        # Kode untuk mengirim perintah ke perangkat IoT
        print(f"Sending command: {command} to {self.name}")


class ATM(Device):
    def __init__(self, name, topic):
        super().__init__(name, topic)

    def withdraw_money(self, amount):
        # Kode untuk melakukan penarikan uang
        print(f"{self.name} is withdrawing money: {amount}")

    def check_balance(self):
        # Kode untuk memeriksa saldo
        print(f"{self.name} is checking balance")


class CreditCardTerminal(Device):
    def __init__(self, name, topic):
        super().__init__(name, topic)

    def read_card(self):
        # Kode untuk membaca kartu kredit
        return "1234567890123456"

    def charge_card(self, amount):
        # Kode untuk mengisi kartu kredit
        print(f"{self.name} is charging card: {amount}")


class Sensor(Device):
    def __init__(self, name, topic):
        super().__init__(name, topic)

    def read_sensor(self):
        # Kode untuk membaca data sensor
        return random.randint(0, 100)
