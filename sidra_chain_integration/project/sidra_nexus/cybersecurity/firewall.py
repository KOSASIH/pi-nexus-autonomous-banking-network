# cybersecurity/firewall.py
import iptables

# Set up a firewall rule
rule = iptables.Rule()
rule.src = '192.168.1.100'
rule.dst = '192.168.1.200'
rule.protocol = 'tcp'
rule.target = iptables.Target()
