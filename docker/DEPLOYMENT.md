# TTD Formal Deployment

MSST runs as a single-host Compose service on `ttd-edge`. Formal builds use the
shared `docker_ci_cd` role to publish only an immutable `h-*` image, then install
the resolved Compose definition at `/opt/msst-webui/compose.yaml`.

Run through the registered TTD wrapper so the shared role is available:

```bash
scripts/ttd-ansible-playbook --exec ./deploy.sh verify
scripts/ttd-ansible-playbook --exec ./deploy.sh build
scripts/ttd-ansible-playbook --exec ./deploy.sh formal
```

The formal lane preserves `/TTD`, model/data mounts, GPU 2, port 8662 and the
internal `http://msst` Caddy route. It does not publish or move `latest`.

Before deployment, record the current container image ID and Compose file. The
Ansible copy task creates a remote backup when the Compose definition changes;
rollback restores that file and reapplies it with `docker compose up -d
--no-build` after confirming the referenced prior image is available.
