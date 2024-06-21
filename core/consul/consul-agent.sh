// consul-agent.sh
#!/bin/bash

CONSUL_AGENT_CONFIG=/etc/consul/consul-agent.json
CONSUL_AGENT_LOG=/var/log/consul-agent.log

systemctl start consul-agent

consul agent -config=$CONSUL_AGENT_CONFIG -log-level=INFO -log-file=$CONSUL_AGENT_LOG
