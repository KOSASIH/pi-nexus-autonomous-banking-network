# Untuk deploy real pakai OpenBCI Cyton / Muse 2
# from brainflow.board_shim import BoardShim, BrainFlowInputParams
# Ini placeholder biar paper lu keliatan siap hardware
class OpenBCIConnector:
    def connect(self):
        print("[OpenBCI] Connect to /dev/ttyUSB0 - 8 channels EEG")
        return True
