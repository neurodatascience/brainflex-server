# brainflex-server

Server for orchestrating brain FL experiments

## Running the VPN server and the Fed-BioMed researcher

The server runs with Docker Compose as three containers:

- `vpn-server`: a WireGuard server that the nodes and the researcher connect to, with Fed-BioMed's `configure_peer` tool. Files are in `vpnserver_config/`.
- `researcher-vpn`: a WireGuard client for the researcher. Files are in `researcher_vpn_config/`.
- `researcher`: the published [`fedbiomed/researcher`](https://hub.docker.com/r/fedbiomed/researcher) image, using the network of the `researcher-vpn` container to reach the nodes, with Jupyter on <http://localhost:8888>. Files are in `fbm-researcher/`.

### Requirements

- Docker with the Compose plugin (`docker compose`)
- Inbound UDP access to port 51820, for the nodes to reach the VPN server

### Setup

1. Create the `.env` file and set `CONTAINER_UID`/`CONTAINER_GID` to the output of `id -u`/`id -g`:

   ```bash
   cp .env.example .env
   ```

2. Create the VPN server configuration. Set `VPN_SERVER_PUBLIC_ADDR` to the public IP address or DNS name of this machine.

   ```bash
   cp vpnserver_config/config.env.sample vpnserver_config/config.env
   ```

3. Build and start the VPN server:

   ```bash
   docker compose up -d --build vpn-server
   ```

4. Create the researcher peer and copy its configuration to the researcher VPN client. The first researcher gets the VPN address `10.222.0.2`, which is the default researcher address of the nodes:

   ```bash
   docker compose exec vpn-server configure_peer genconf researcher researcher1
   cp vpnserver_config/config_peers/researcher/researcher1/config.env researcher_vpn_config/
   ```

5. Start the researcher:

   ```bash
   docker compose up -d --build
   ```

6. Register the public key of the researcher on the VPN server:

   ```bash
   docker compose exec vpn-server configure_peer add researcher researcher1 "$(docker compose exec -T researcher-vpn wg show wg0 public-key)"
   ```

7. Check that all containers are running and that the researcher can reach the VPN server:

   ```bash
   scripts/vpn_status
   ```

### Adding a node

1. Create the node peer:

   ```bash
   docker compose exec vpn-server configure_peer genconf node <NODE_ID>
   ```

2. Send `vpnserver_config/config_peers/node/<NODE_ID>/config.env` to the site running the node via a secure channel.

3. Once the site has started its node, register the WireGuard public key it sends back:

   ```bash
   docker compose exec vpn-server configure_peer add node <NODE_ID> <PUBLIC_KEY>
   ```

4. List the peers and their keys:

   ```bash
   docker compose exec vpn-server configure_peer list
   ```

### Running the experiments

The experiment script can be invoked via e.g.:

```bash
docker compose exec -u fedbiomed researcher python /brainflex-server/scripts/run_experiments.py --help
```

The output directory should be `fbm-researcher/results` inside the container.

### Stopping

```bash
docker compose down
```

 To start from scratch, delete everything in `vpnserver_config/`, `researcher_vpn_config/` and `fbm-researcher/` except `.gitkeep` and `config.env.sample`.

## Development

This project uses [uv](https://docs.astral.sh/uv/).

### Installation

```bash
uv sync
```

### Git hooks via `prek`

```bash
uv run prek install
uv run prek run -a
```

### Tests

```bash
uv run pytest
```
