#!/bin/bash

# Exit immediately if any command returns a non-zero status
set -e

# Enable nounset to treat unset variables as an error
set -o nounset

# Enable pipefail to monitor the exit state of pipelines
set -o pipefail

# Validate user input
read -p "Enter the directory to scan for security vulnerabilities: " dir
if [[ ! -d "$dir" ]]; then
  echo "Invalid directory"
  exit 1
fi

# Check for common security vulnerabilities
echo "Checking for world-writable files..."
find "$dir" -type d \( -perm -002 -a ! -perm -100 \) -or \( -perm -007 -a ! -perm -100 \) -print0 | xargs -0 chmod o-w

echo "Checking for SUID/SGID files..."
find "$dir" \( -perm -4000 -o -perm -2000 \) -print0 | xargs -0 ls -l

echo "Checking for files with permissive permissions..."
find "$dir" \( -perm -0002 -o -perm -0004 -o -perm -0007 \) -print0 | xargs -0 ls -l

echo "Checking for open ports..."
nmap -sT -O "$(hostname)"

echo "Checking for outdated packages..."
apt-get update && apt-get upgrade -y

echo "Checking for unintended network connections..."
lsof -i

echo "Checking for hidden files and directories..."
find "$dir" -name '.*'

echo "Checking for process listening on privileged ports..."
netstat -tuln | grep ':[0-9]\{1,5\}' | awk '{print $4}' | cut -d: -f1 | grep -E '^(0|7|9|13|21|25|42|43|53|79|87|110|111|119|135|139|143|179|389|443|445|543|544|546|547|548|554|555|556|563|631|636|989|990|993|995|1433|1434|1720|1723|1741|1755|1900|2000|2001|2049|2105|2107|2717|3000|3128|3306|3389|3777|4899|5000|5060|5150|5353|5354|5355|5432|5672|5900|5985|5986|6000|6646|7000|7001|7070|7510|8000|8001|8008|8080|8081|8443|8888|9000|9090|9200|. 
================. 
================
