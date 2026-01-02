---
description: How to develop locally and deploy to the Raspberry Pi
---

# Remote Development Workflow

To sync your code and run tests on the Raspberry Pi:

1. **Ensure SSH access**: You must have SSH access to `pi@192.168.178.48`. Key-based authentication is highly recommended to avoid password prompts.
2. **Run the deployment script**:
   ```bash
   ./deploy_to_pi.sh
   ```

## What happens:
- **Local**: `rsync` sends your code (minus `build/`, `.git/`, etc.) to `/home/pi/CoralEdgeTpu`.
- **Remote**: SSH triggers `remote_test.sh` on the Pi.
- **Remote**: `remote_test.sh` ensures the `build/` directory exists, runs CMake if needed, builds the app with `make -j2`, and starts the `detector`.
