from scapy.all import *

def block_ip(ip):
    ip_block = IP(dst=ip) / ICMP(type=13, code=1)
    send(ip_block, verbose=0)

def mitigate_threats(pcap_file):
    packets = rdpcap(pcap_file)
    blocked_ips = set()
    for packet in packets:
        if packet.haslayer(TCP):
            if packet[TCP].flags & 2 and packet[IP].src not in blocked_ips:
                print(f"Blocking SYN flood from {packet[IP].src}")
                block_ip(packet[IP].src)
                blocked_ips.add(packet[IP].src)
            elif packet[TCP].flags & 18 and packet[IP].src not in blocked_ips:
                print(f"Blocking RST flood from {packet[IP].src}")
                block_ip(packet[IP].src)
                blocked_ips.add(packet[IP].src)
        elif packet.haslayer(ICMP):
            if packet[ICMP].type == 8 and packet[IP].src not in blocked_ips:
                print(f"Blocking ping flood from {packet[IP].src}")
                block_ip(packet[IP].src)
                blocked_ips.add(packet[IP].src)

if __name__ == "__main__**:
    mitigate_threats("example.pcap")
