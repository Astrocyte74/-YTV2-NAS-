# WireGuard Remote Access (ASUSTOR AS5304T)

This captures the exact steps we used to reach the NAS remotely while keeping everything (dashboard, APIs, SSH) behaving like we’re on the home LAN.

## 1. Enable WireGuard on the NAS

1. Open **VPN Server** in ADM.
2. Go to **Settings → WireGuard**.
3. Tick **Enable** and click **Generate Keypair**.
4. Leave the defaults (`Address 10.0.4.1/24`, `Listen port 51820`) and click **Apply**.

## 2. Create a WireGuard peer for each client

1. In **VPN Server**, open **Privilege → WireGuard Peer**.
2. Click **Create**.
3. Paste the client’s **public key** (generated on the client; see section 4) and name the peer (e.g. `macbook`).
4. Set **Allowed IPs** to a unique tunnel address such as `10.0.4.2/32`.
5. Leave **Persistent keepalive** at 25 seconds and click **OK**, then **Apply**.

Repeat for other devices so each peer has its own keypair/IP.

## 3. Forward UDP 51820 on the router (Eero)

**Settings → Network Settings → Reservations & Port Forwarding**

- Reserve the NAS IP `192.168.4.54`.
- Add a port forward named `WireGuard`:
  - External port: 51820
  - Internal IP: 192.168.4.54
  - Internal port: 51820
  - Protocol: **UDP**

## 4. Configure the client (macOS example)

1. Install the WireGuard app.
2. **Add tunnel → Create from scratch**.
3. The interface section auto-generates `PrivateKey` and `PublicKey`. Copy the public key and paste it into the NAS peer created above.
4. Fill in the tunnel manually:

```ini
[Interface]
PrivateKey = <generated private key>
Address = 10.0.4.2/32
DNS = 1.1.1.1

[Peer]
PublicKey = <NAS public key from Settings → WireGuard>
Endpoint = 24.66.251.193:51820
AllowedIPs = 10.0.4.0/24, 192.168.4.0/24
PersistentKeepalive = 25
```

Use fresh addresses (`10.0.4.3/32`, etc.) for other peers.

## 5. Activate and verify

1. In **VPN Server → Overview**, toggle **WireGuard** on.
2. On the client, activate the tunnel (`asustor`). The status dot should turn green.
3. Test:

```bash
ping 192.168.4.54
curl http://192.168.4.54:6452/api/metrics
```

Success means the NAS LAN is reachable while internet stays local.

## 6. SSH over the VPN

With the tunnel active you can SSH via the LAN IP:

```bash
ssh -p 22 mcdarby2024@192.168.4.54
```

(Port 1515 to the public IP still works if needed.)

## 7. Adding more devices later

- Create another peer on the NAS (Privilege → WireGuard Peer → Create).
- Generate a new keypair on the device’s WireGuard app.
- Paste that public key into the NAS peer entry; assign the next tunnel IP.
- Import/save the config on the device and connect.

## 8. Troubleshooting checklist

- **Tunnel won’t connect**: confirm WireGuard is enabled in VPN Server and the UDP 51820 forward still points to 192.168.4.54.
- **Tunnel says Active but NAS unreachable**: double-check the NAS public key in the client `[Peer]` section and ensure `AllowedIPs` includes `10.0.4.0/24, 192.168.4.0/24`.
- **Internet dies when VPN active**: set `AllowedIPs` to the two LAN subnets (not `0.0.0.0/0`).
- **Need full tunnel later**: enable IP forwarding + NAT on the NAS, then switch `AllowedIPs` to `0.0.0.0/0, ::/0` and set `DNS` to 192.168.4.1 (or another resolver).

Once connected, the dashboard, `/api/metrics`, `/api/report-events`, `/api/reprocess`, and SSH behave exactly as if you were on the home network.
