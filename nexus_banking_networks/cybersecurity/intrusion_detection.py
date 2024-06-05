import scapy


class IntrusionDetection:

    def __init__(self, network_interface):
        self.network_interface = network_interface
        self.sniffer = scapy.sniff(iface=self.network_interface)

    def detect_intrusions(self):
        # Detect intrusions using Scapy
        for packet in self.sniffer:
            if packet.haslayer(scapy.TCP) and packet.dport == 22:
                print("SSH connection detected!")
