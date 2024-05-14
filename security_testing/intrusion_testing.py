from scapy.all import *
import sys

def intrusion_tester(ip, ports):
    """Test a host for open ports."""
    for port in ports:
        print(f"Testing port {port} on {ip}...")
        packet = IP(dst=ip)/TCP(sport=RandShort(), dport=port, flags="S")
        response = sr1(packet, timeout=1, verbose=0)
        if response is not None:
            if response.haslayer(TCP):
                if response.getlayer(TCP).flags == 0x12:
                    print(f"Port {port} is open on {ip}")
                    sys.exit(1)

if __name__ == "__main__":
    ip = input("Enter the IP address to test: ")
    ports = [21, 22, 25, 80, 443, 3389]
    intrusion_tester(ip, ports)
