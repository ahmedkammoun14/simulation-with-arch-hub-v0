import asyncio
import logging
import random
from datetime import datetime, timezone

import uvicorn
from fastapi import FastAPI, Request

# Configuration des profils de simulation
VM_PROFILES = {
    "vm1": {"port": 8101, "rtt": (10, 20), "cpu": (40, 65), "ram": (50, 70)},
    "vm2": {"port": 8102, "rtt": (40, 60), "cpu": (80, 95), "ram": (75, 90)},
    "vm3": {"port": 8103, "rtt": (5, 15), "cpu": (30, 50), "ram": (40, 60)},
    "vm4": {"port": 8104, "rtt": (20, 35), "cpu": (60, 80), "ram": (65, 80)},
}

# Configuration du logging
logging.basicConfig(
    level=logging.INFO,
    format="%(message)s"
)
logger = logging.getLogger("vm_simulator")

def create_vm_app(vm_id: str, profile: dict) -> FastAPI:
    """Crée une instance FastAPI configurée pour une VM spécifique."""
    app = FastAPI(title=f"Simulator {vm_id}")

    @app.get("/health")
    async def health():
        val = random.uniform(*profile["rtt"])
        ts = datetime.now(timezone.utc).isoformat()
        logger.info(f"[VM-{vm_id}] GET /health → response_time_ms={val:.1f}ms timestamp={ts}")
        return {
            "vm_id": vm_id,
            "status": "ok",
            "response_time_ms": val,
            "timestamp": ts
        }

    @app.get("/metrics")
    async def metrics():
        cpu = random.uniform(*profile["cpu"])
        ram = random.uniform(*profile["ram"])
        ts = datetime.now(timezone.utc).isoformat()
        logger.info(f"[VM-{vm_id}] GET /metrics → cpu={cpu:.1f}% ram={ram:.1f}% timestamp={ts}")
        return {
            "vm_id": vm_id,
            "cpu_usage": cpu,
            "ram_usage": ram,
            "timestamp": ts
        }

    return app

async def run_vm_server(vm_id: str, profile: dict):
    """Lance un serveur Uvicorn pour une VM donnée."""
    app = create_vm_app(vm_id, profile)
    config = uvicorn.Config(app, host="0.0.0.0", port=profile["port"], log_level="error")
    server = uvicorn.Server(config)
    await server.serve()

async def main():
    """Lance toutes les VMs en parallèle."""
    logger.info(f"Démarrage de {len(VM_PROFILES)} simulateurs de VM...")
    tasks = [run_vm_server(vm_id, profile) for vm_id, profile in VM_PROFILES.items()]
    await asyncio.gather(*tasks)

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Arrêt des simulateurs.")
