FROM ipfs/go-ipfs:latest

# Copy the swarm key
COPY swarm.key /data/ipfs/swarm.key

# Copy a startup script
COPY start-ipfs.sh /usr/local/bin/
RUN chmod +x /usr/local/bin/start-ipfs.sh

# Use the script as entrypoint
ENTRYPOINT ["/usr/local/bin/start-ipfs.sh"]