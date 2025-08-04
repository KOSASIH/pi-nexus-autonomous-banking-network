from scapy.all import *


def detect_threats(pcap_file):
    packets = rdpcap(pcap_file)
    for packet in packets:
        if packet.haslayer(TCP):
            if packet[TCP].flags & 2:
                print(f"SYN flood detected from {packet[IP].src}")
            elif packet[TCP].flags & 18:
                print(f"RST flood detected from {packet[IP].src}")
        elif packet.haslayer(ICMP):
            if packet[ICMP].type == 8:
                print(f"Ping flood detected from {packet[IP].src}")


if __name__ == "__main__":
    detect_threats("example.pcap")
