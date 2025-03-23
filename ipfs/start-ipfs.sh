#!/bin/sh
# Initialize IPFS if not already initialized
if [ ! -f /data/ipfs/config ]; then
    echo "Initializing IPFS..."
    ipfs init --profile=server

    # Configure for public network
    echo "Configuring for public IPFS network..."
    
    # Reset bootstrap nodes to defaults
    ipfs bootstrap rm --all
    ipfs bootstrap add --default
fi

# Configure IPFS to listen on all interfaces
ipfs config Addresses.API /ip4/0.0.0.0/tcp/5001
ipfs config Addresses.Gateway /ip4/0.0.0.0/tcp/8080
ipfs config --json Addresses.Swarm '["/ip4/0.0.0.0/tcp/4001", "/ip4/0.0.0.0/tcp/8081/ws"]'

# Increase connection limits for better network connectivity
ipfs config --json Swarm.ConnMgr.HighWater 200
ipfs config --json Swarm.ConnMgr.LowWater 100
ipfs config --json Swarm.ConnMgr.GracePeriod "30s"

# Configure for better performance
ipfs config --json Datastore.BloomFilterSize 1048576
ipfs config --json Reprovider.Interval "12h"

# Optimize gateway for reading content
ipfs config --json Gateway.RootRedirect ""
ipfs config --json Gateway.Writable false
ipfs config --json Gateway.PathPrefixes []
ipfs config --json Gateway.APICommands []

# Enable CORS for API access
ipfs config --json API.HTTPHeaders.Access-Control-Allow-Origin '["*"]'
ipfs config --json API.HTTPHeaders.Access-Control-Allow-Methods '["GET", "POST"]'

echo "Starting IPFS daemon with routing enabled..."
# Start the daemon with routing enabled for public DHT
exec ipfs daemon --migrate --routing=dhtclient