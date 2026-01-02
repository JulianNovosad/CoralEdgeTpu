#!/bin/bash
# Deploy script with password

# Use SSH_ASKPASS to provide password
export SSH_ASKPASS_REQUIRE=force
export SSH_ASKPASS="/home/julian/CoralEdgeTpu/.ssh_pass.sh"
export DISPLAY=:0

# Create password script
cat > /home/julian/CoralEdgeTpu/.ssh_pass.sh << 'EOF'
#!/bin/bash
echo "pi"
EOF
chmod +x /home/julian/CoralEdgeTpu/.ssh_pass.sh

# Rsync files
rsync -avz --exclude 'build/' --exclude '.git/' --exclude '*.o' -e "ssh -o StrictHostKeyChecking=no" ./ pi@192.168.178.48:/home/pi/CoralEdgeTpu/

# Run remote build and test
ssh -o StrictHostKeyChecking=no pi@192.168.178.48 'cd /home/pi/CoralEdgeTpu && ./remote_test.sh'

# Cleanup
rm -f /home/julian/CoralEdgeTpu/.ssh_pass.sh
