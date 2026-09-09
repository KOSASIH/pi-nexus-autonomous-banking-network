"""Network threat mitigation helpers built on Scapy.

The module inspects captured traffic for classic flood patterns (SYN, RST and
ping floods) and blocks offending source addresses.
"""

import scapy.all as scapy


def block_ip(ip):
    """Send an ICMPv4 Unreachable to the given host, dropping its traffic.

    Args:
        ip: IPv4 address to block.
    """
    ip_block = scapy.IP(dst=ip) / scapy.ICMP(type=13, code=1)
    scapy.send(ip_block, verbose=0)


def mitigate_threats(pcap_file):
    """Scan a pcap for flood patterns and block offending sources.

    Args:
        pcap_file: path to a packet capture readable by Scapy.

    Returns:
        A set of IP addresses that were blocked.
    """
    packets = scapy.rdpcap(pcap_file)
    blocked_ips = set()
    for packet in packets:
        if not packet.haslayer(scapy.IP):
            continue
        src = packet[scapy.IP].src
        if packet.haslayer(scapy.TCP):
            if packet[scapy.TCP].flags & 2 and src not in blocked_ips:
                print("Blocking SYN flood from", src)
                block_ip(src)
                blocked_ips.add(src)
            elif packet[scapy.TCP].flags & 18 and src not in blocked_ips:
                print("Blocking RST flood from", src)
                block_ip(src)
                blocked_ips.add(src)
        elif packet.haslayer(scapy.ICMP):
            if packet[scapy.ICMP].type == 8 and src not in blocked_ips:
                print("Blocking ping flood from", src)
                block_ip(src)
                blocked_ips.add(src)
    return blocked_ips


if __name__ == "__main__":
    mitigate_threats("example.pcap")
