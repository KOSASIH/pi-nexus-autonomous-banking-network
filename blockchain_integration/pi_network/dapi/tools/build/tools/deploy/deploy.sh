# tools/deploy/deploy.sh
#!/bin/bash

# Deploy the project
echo "Deploying the project..."
ssh remote_server "cd ~/project && git pull && make install"
