# Deployment Examples

This document describes how to run `qwen3-engine-daemon` as a system service on macOS (`launchd`) and Linux (`systemd`).

Before installing either service, replace the model path placeholder in the example files:

- `/path/to/your/model.gguf` → path to your actual GGUF model file.

The daemon binary is assumed to be built at:

```text
/Users/wangyang/metal_demo/qwen3-engine-daemon
```

---

## macOS: launchd

Example file: [`examples/launchd/qwen3-engine-daemon.plist`](../examples/launchd/qwen3-engine-daemon.plist)

### Install as a user agent

```bash
# Copy the plist to your personal LaunchAgents directory
cp /Users/wangyang/metal_demo/examples/launchd/qwen3-engine-daemon.plist \
   ~/Library/LaunchAgents/com.qwen3-engine.daemon.plist

# Load the job
launchctl load ~/Library/LaunchAgents/com.qwen3-engine.daemon.plist

# Start the job
launchctl start com.qwen3-engine.daemon

# Check that it is loaded/running
launchctl list | grep com.qwen3-engine.daemon
```

### Install as a system daemon (runs as root)

```bash
sudo cp /Users/wangyang/metal_demo/examples/launchd/qwen3-engine-daemon.plist \
        /Library/LaunchDaemons/com.qwen3-engine.daemon.plist
sudo launchctl load /Library/LaunchDaemons/com.qwen3-engine.daemon.plist
sudo launchctl start com.qwen3-engine.daemon
```

### View logs

`StandardOutPath` and `StandardErrorPath` are both set to:

```text
/tmp/qwen3-engine-daemon.log
```

```bash
tail -f /tmp/qwen3-engine-daemon.log
```

### Unload / stop

```bash
launchctl stop com.qwen3-engine.daemon
launchctl unload ~/Library/LaunchAgents/com.qwen3-engine.daemon.plist
```

---

## Linux: systemd

Example file: [`examples/systemd/qwen3-engine-daemon.service`](../examples/systemd/qwen3-engine-daemon.service)

### Install the service

```bash
sudo cp /Users/wangyang/metal_demo/examples/systemd/qwen3-engine-daemon.service \
        /etc/systemd/system/qwen3-engine-daemon.service

sudo systemctl daemon-reload
sudo systemctl enable qwen3-engine-daemon.service
sudo systemctl start qwen3-engine-daemon.service
```

### Check status and logs

```bash
sudo systemctl status qwen3-engine-daemon.service
sudo journalctl -u qwen3-engine-daemon -f
```

### Restart / stop

```bash
sudo systemctl restart qwen3-engine-daemon.service
sudo systemctl stop qwen3-engine-daemon.service
```

---

## Notes

- Make sure the socket path used by `-s` matches the path expected by clients.
- Adjust `-c` (context size), KV provider flags (`-k`, `-K`), and daemon ID (`-i`) in the service files to suit your deployment.
- On macOS, user agents run only while the user is logged in; use `/Library/LaunchDaemons` for headless operation.
