#!/bin/sh
# Initialize IPFS if not already initialized
if [ ! -f /data/ipfs/config ]; then
    echo "Initializing IPFS..."
    ipfs init
fi

# Ensure the swarm key is in place
if [ -f /data/ipfs/swarm.key ]; then
  echo "Swarm key found. Configuring private swarm."
else
  echo "Swarm key not found. Exiting."
  exit 1
fi

ipfs bootstrap rm --all

# Configure IPFS to listen on all interfaces
ipfs config Addresses.API /ip4/0.0.0.0/tcp/5002
ipfs config Addresses.Gateway /ip4/0.0.0.0/tcp/8081
# ipfs config Addresses.Swarm /ip4/0.0.0.0/tcp/4001
# Configure Swarm to listen on all interfaces (use JSON format for array)
ipfs config Addresses.Swarm --json '[
  "/ip4/0.0.0.0/tcp/4002",
  "/ip6/::/tcp/4002"
]'

# Announce the public IP addresss
ipfs config --json Addresses.Announce "[\"/ip4/129.74.152.201/tcp/4002\"]"

# enable DHT routing
ipfs config Routing.Type dhtclient

# Disable AutoTLS to avoid conflicts with private networking
ipfs config --json AutoTLS.Enabled false

# Disable WebSocket transport for private networks
ipfs config --json Swarm.Transports.Network.Websocket true

# Disable TCP multiplexing if necessary
export LIBP2P_TCP_MUX=true

# Start the daemon
exec ipfs daemon --migrate=true