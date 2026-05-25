from __future__ import annotations

import logging
import threading
import sqlite3
import json
import time
import random
import statistics
import re
import functools
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import List, Dict, Optional, Any

# Bibliothèques pour les APIs et la visualisation
from flask import Flask, request, jsonify
import requests
import matplotlib.pyplot as plt
import colorama
from colorama import Fore, Back, Style

# Initialisation Colorama
colorama.init(autoreset=True)

# Désactivation des logs Werkzeug
log = logging.getLogger('werkzeug')
log.setLevel(logging.ERROR)

# Configuration du logging personnalisé
class ColoredFormatter(logging.Formatter):
    def format(self, record):
        level_color = {
            logging.INFO: Fore.CYAN,
            logging.WARNING: Fore.YELLOW,
            logging.ERROR: Fore.RED,
            logging.CRITICAL: Back.RED + Fore.WHITE
        }.get(record.levelno, Fore.WHITE)
        
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return f"{Fore.BLACK + Style.BRIGHT}{timestamp}{Style.RESET_ALL} | {level_color}{record.levelname:8}{Style.RESET_ALL} | {Style.BRIGHT}{record.getMessage()}{Style.RESET_ALL}"

handler = logging.StreamHandler()
handler.setFormatter(ColoredFormatter())
logging.root.handlers = [handler]
logging.root.setLevel(logging.INFO)
logger = logging.getLogger("Orchestrator")

def log_step(step_name: str, direction: str, data: Any):
    import json as _json
    # Convertir en JSON lisible
    try:
        if isinstance(data, (dict, list)):
            data_lines = _json.dumps(data, indent=2, ensure_ascii=False).split('\n')
        else:
            data_lines = [str(data)]
    except:
        data_lines = [str(data)]
    width = 60
    print(f"\n{Fore.BLUE}┌{'─'*width}┐")
    print(f"{Fore.BLUE}│  {Fore.YELLOW}➤ ÉTAPE : {Style.BRIGHT}{step_name}")
    print(f"{Fore.BLUE}│  {Fore.CYAN}📤 {direction}")
    print(f"{Fore.BLUE}│  {Fore.WHITE}📦 PAYLOAD :")
    for line in data_lines:
        print(f"{Fore.BLUE}│    {Fore.MAGENTA}{line}")
    print(f"{Fore.BLUE}└{'─'*width}┘")

@dataclass
class Config:
    """Paramètres globaux de l'orchestrateur (Ports, IPs, Seuils)."""
    CORE_PORT: int = 8000
    LATENCY_PORT: int = 8010
    ML_PREDICTOR_PORT: int = 8011
    COLLECTOR_PORT: int = 8012
    DECISION_PORT: int = 8013
    INTENT_PORT: int = 8014
    CONFIG_PORT: int = 8015
    OBSERVABILITY_PORT: int = 8016
    DATABASE_PORT: int = 8020
    HISTORY_LOADER_PORT: int = 8021
    METRICS_MANAGER_PORT: int = 8022
    
    INTENT_ENGINE_URL: str = "http://localhost:11434/api/chat"
    ML_RTT_URL: str = "http://localhost:5001/predict"
    ML_CPU_URL: str = "http://localhost:5002/predict"
    ML_RAM_URL: str = "http://localhost:5003/predict"
    
    VM_LIST: List[str] = field(default_factory=lambda: ["vm1", "vm2", "vm3", "vm4"])
    VM_PORTS: Dict[str, int] = field(default_factory=lambda: {
        "vm1": 8101, "vm2": 8102, "vm3": 8103, "vm4": 8104
    })
    
    DEFAULT_LATENCY_THRESHOLD: float = 50.0
    DEFAULT_CPU_THRESHOLD: float = 75.0
    DEFAULT_RAM_THRESHOLD: float = 80.0
    
    COLLECTION_INTERVAL: int = 5
    HISTORY_WINDOW: int = 10
    DB_NAME: str = "orchestrator.db"

    # Master Cloud Config
    MASTER_URL: str = "https://master-cloud/api/v1/migrate"
    MASTER_TOKEN: str = "changeme"
    MASTER_TIMEOUT: int = 10

# -----------------------------------------------------------------------------
# UTILS
# -----------------------------------------------------------------------------

def safe_call(default_val: Any, func_name: str):
    """Décorateur qui catch toutes les exceptions et les log sans crasher."""
    def decorator(func):
        @functools.wraps(func)
        def wrapper(*args, **kwargs):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                logger.error(f"[SafeCall] Error in {func_name}: {e}")
                return default_val
        return wrapper
    return decorator

# -----------------------------------------------------------------------------
# SPOKES (SERVICES PÉRIPHÉRIQUES)
# -----------------------------------------------------------------------------

class LatencyManagerSpoke:
    def __init__(self, config: Config):
        self.config = config
        self.core: Optional['OrchestratorCore'] = None
        self.app = Flask(f"{__name__}_latency")

    def _setup_routes(self):
        @self.app.route('/rtt', methods=['POST'])
        def receive_rtt():
            data = request.json
            if not data or "measurements" not in data or not data["measurements"]:
                return jsonify({"error": "Invalid payload"}), 400
            if self.core:
                self.core.last_real_data_ts = time.time()
                measurements = data["measurements"]
                decision = self.core.run_enhanced_flow(measurements) if self.core.mode == "enhanced" else self.core.run_classic_flow(measurements)
                return jsonify({"status": "received", "decision": decision}), 200
            return jsonify({"status": "core_not_ready"}), 503

    def start_api(self):
        self._setup_routes()
        threading.Thread(target=self.app.run, kwargs={'host': '0.0.0.0', 'port': self.config.LATENCY_PORT, 'threaded': True, 'use_reloader': False}, daemon=True).start()
        logger.info(f"[LatencyManager] API started on port {self.config.LATENCY_PORT}")

class IntentManagerSpoke:
    def __init__(self, config: Config):
        self.config = config
        self.core: Optional['OrchestratorCore'] = None
        self.app = Flask(f"{__name__}_intent")

    def _setup_routes(self):
        @self.app.route('/intent', methods=['POST'])
        def receive_intent():
            data = request.json
            if not data or "intention" not in data or not data["intention"]:
                return jsonify({"error": "Empty intention"}), 400
            if self.core:
                self.core.set_user_intent(data["intention"])
                return jsonify({"status": "received", "intent_id": data.get("intent_id", "unknown"), "slos": self.core.current_slos}), 200
            return jsonify({"status": "core_not_ready"}), 503

    def start_api(self):
        self._setup_routes()
        threading.Thread(target=self.app.run, kwargs={'host': '0.0.0.0', 'port': self.config.INTENT_PORT, 'threaded': True, 'use_reloader': False}, daemon=True).start()
        logger.info(f"[IntentManager] API started on port {self.config.INTENT_PORT}")

    def query_intent_engine(self, text: str) -> List[Dict]:
        try:
            payload = {
                "model": "qwen2.5",
                "messages": [
                    {
                        "role": "system",
                        "content": (
                            "You are an assistant that MUST respond with ONLY a JSON array and nothing else. "
                            "Do not include explanations, markdown, or any text outside the JSON array. "
                            "Extract numeric threshold values directly from the user's intention text when present. "
                            "Only include SLO objects for metrics explicitly mentioned. "
                            "Each item must have keys: metric, operator, threshold, unit. "
                            "Allowed metrics: \"latency\", \"cpu_usage\", \"ram_usage\". "
                            "Allowed operators: \"<\", \"<=\", \">\", \">=\". "
                            "Unit must be \"ms\" for latency and \"%\" for cpu/ram. "
                            "Example: [{\"metric\":\"latency\",\"operator\":\"<\",\"threshold\":20,\"unit\":\"ms\"}]"
                        )
                    },
                    {"role": "user", "content": text}
                ],
                "stream": False
            }
            resp = requests.post(self.config.INTENT_ENGINE_URL, json=payload, timeout=30)
            llm_text = resp.json().get("message", {}).get("content", "").lower()
            return self._parse_slos(llm_text)
        except Exception as e:
            logger.warning(f"[Intent] LLM Error: {e}. Trying direct regex parsing.")
            return self._parse_slos(text)

    def _parse_slos(self, text: str) -> List[Dict]:
        # Attempt JSON parsing first
        try:
            start = text.find('[')
            end = text.rfind(']')
            if start != -1 and end != -1 and end > start:
                json_part = text[start:end+1]
                slos = json.loads(json_part)
                if isinstance(slos, list) and len(slos) > 0:
                    return slos
        except Exception:
            pass

        # Fallback to Regex parsing
        slos = []
        patterns = {"latency": r"latency\s*<\s*(\d+)", "cpu_usage": r"cpu\s*<\s*(\d+)", "ram_usage": r"ram\s*<\s*(\d+)"}
        for metric, regex in patterns.items():
            match = re.search(regex, text)
            if match:
                slos.append({"metric": metric, "operator": "<", "threshold": float(match.group(1)), "unit": "ms" if metric == "latency" else "%"})
        return slos if slos else [
            {"metric": "latency", "operator": "<", "threshold": 50, "unit": "ms"},
            {"metric": "cpu_usage", "operator": "<", "threshold": 75, "unit": "%"},
            {"metric": "ram_usage", "operator": "<", "threshold": 80, "unit": "%"}
        ]

class MetricsManagerSpoke:
    def __init__(self, config: Config): self.config = config
    def analyze_slos(self, slos: List[Dict]) -> List[str]:
        needed = []
        for slo in slos:
            if slo["metric"] in ["cpu_usage", "ram_usage"]: needed.append(slo["metric"])
        return list(set(needed))

class CollectorSpoke:
    def __init__(self, config: Config): self.config = config

    @safe_call({}, "CollectorSpoke.collect_vm_metrics")
    def collect_vm_metrics(self, vm_id: str) -> Dict:
        return {"vm_id": vm_id, "cpu_usage": random.uniform(20, 95), "ram_usage": random.uniform(30, 90)}

class MLPredictorSpoke:
    def __init__(self, config: Config): 
        self.config = config

    def get_prediction(self, current_val: float, history: List[float]) -> List[float]:
        """Classic RTT prediction with internal fallback (API V2 format)."""
        try:
            normalized = current_val / 100.0
            resp = requests.get(
                self.config.ML_RTT_URL,
                params={"input_data": normalized},
                timeout=10
            )
            raw = resp.json()["prediction"]
            # Parser la string numpy : "[0.32 0.33 ...]"
            import re as _re
            values = [float(x) for x in _re.findall(r"[\d.]+", raw)]
            return [v * 100 for v in values]
        except Exception:
            logger.warning(f"[ML] API latency indisponible → simulation activée")
            return [current_val * (1.05 ** i) for i in range(1, 6)]

    def get_enhanced_prediction(self, metric: str, current_val: float, history: List[float]) -> List[float]:
        """Enhanced prediction with internal fallback (API V2 format)."""
        urls = {"latency": self.config.ML_RTT_URL, "cpu_usage": self.config.ML_CPU_URL, "ram_usage": self.config.ML_RAM_URL}
        factors = {"latency": 1.05, "cpu_usage": 1.03, "ram_usage": 1.02}
        try:
            normalized = current_val / 100.0
            resp = requests.get(
                urls.get(metric),
                params={"input_data": normalized},
                timeout=10
            )
            raw = resp.json()["prediction"]
            # Parser la string numpy : "[0.32 0.33 ...]"
            import re as _re
            values = [float(x) for x in _re.findall(r"[\d.]+", raw)]
            return [v * 100 for v in values]
        except Exception:
            logger.warning(f"[ML] API {metric} indisponible → simulation activée")
            f = factors.get(metric, 1.01)
            return [current_val * (f ** i) for i in range(1, 6)]

class DecisionIntelligenceSpoke:
    def __init__(self, config: Config): self.config = config
    def evaluate_decision(self, current_data: List[Dict], predictions_map: Dict[str, List[float]]) -> Dict:
        threshold = self.config.DEFAULT_LATENCY_THRESHOLD
        breached_vm = None
        trigger_entry = None
        for entry in current_data:
            vm_id = entry["vm_id"]
            if entry["rtt_ms"] > threshold or statistics.median(predictions_map.get(vm_id, [])) > threshold:
                breached_vm = vm_id
                trigger_entry = entry
                break
        if breached_vm:
            targets = [e for e in current_data if e["vm_id"] != breached_vm]
            best_target = min(targets, key=lambda x: x["rtt_ms"])
            reason = f"{breached_vm} latency {trigger_entry['rtt_ms']:.1f}ms exceeds threshold {threshold}ms, {best_target['vm_id']} has lowest RTT ({best_target['rtt_ms']:.1f}ms)"
            return {"decision": "migrate", "from_vm": breached_vm, "to_vm": best_target["vm_id"], "reason": reason}
        return {"decision": "stay", "reason": "Nominal"}

    def evaluate_enhanced_decision(self, current_data: List[Dict], predictions_map: Dict[str, Dict[str, List[float]]], slos: List[Dict]) -> Dict:
        for entry in current_data:
            vm_id = entry["vm_id"]
            for slo in slos:
                metric = slo["metric"]; threshold = slo["threshold"]
                current_val = entry.get("rtt_ms", 0) if metric == "latency" else entry.get(metric, 0)
                preds = predictions_map.get(vm_id, {}).get(metric, [])
                median_pred = statistics.median(preds) if preds else 0
                if current_val > threshold or median_pred > threshold:
                    targets = [e for e in current_data if e["vm_id"] != vm_id]
                    best_target = min(targets, key=lambda x: x["rtt_ms"])
                    reason = f"{vm_id} {metric} {current_val:.1f} exceeds SLO threshold {threshold}, {best_target['vm_id']} respects all SLOs"
                    return {"decision": "migrate", "from_vm": vm_id, "to_vm": best_target["vm_id"], "reason": reason}
        return {"decision": "stay", "reason": "All SLOs satisfied"}

class DatabaseSpoke:
    def __init__(self, config: Config): self.config = config

    def init_db(self):
        with sqlite3.connect(self.config.DB_NAME) as conn:
            conn.execute("""CREATE TABLE IF NOT EXISTS metrics (id INTEGER PRIMARY KEY AUTOINCREMENT, vm_id TEXT, rtt_ms REAL, cpu_usage REAL, ram_usage REAL, mode TEXT, timestamp TEXT)""")
            conn.execute("""CREATE TABLE IF NOT EXISTS decisions (id INTEGER PRIMARY KEY AUTOINCREMENT, decision TEXT, from_vm TEXT, to_vm TEXT, reason TEXT, mode TEXT, master_ack INTEGER, timestamp TEXT)""")

    @safe_call(None, "DatabaseSpoke.save_metrics")
    def save_metrics(self, measurements: List[Dict], mode: str):
        ts = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.config.DB_NAME) as conn:
            for m in measurements:
                conn.execute("INSERT INTO metrics (vm_id, rtt_ms, cpu_usage, ram_usage, mode, timestamp) VALUES (?, ?, ?, ?, ?, ?)", (m["vm_id"], m.get("rtt_ms", 0), m.get("cpu_usage", 0), m.get("ram_usage", 0), mode, ts))

    @safe_call(False, "DatabaseSpoke.save_decision")
    def save_decision(self, decision: Dict, mode: str, master_ack: bool):
        ts = datetime.now(timezone.utc).isoformat()
        with sqlite3.connect(self.config.DB_NAME) as conn:
            conn.execute("INSERT INTO decisions (decision, from_vm, to_vm, reason, mode, master_ack, timestamp) VALUES (?, ?, ?, ?, ?, ?, ?)", (decision["decision"], decision.get("from_vm"), decision.get("to_vm"), decision["reason"], mode, 1 if master_ack else 0, ts))
        return True

class HistoryLoaderSpoke:
    def __init__(self, config: Config): self.config = config
    def load_history_window(self, vm_id: str, metric: str, window_size: int) -> List[float]:
        col = "rtt_ms" if metric == "latency" else metric
        with sqlite3.connect(self.config.DB_NAME) as conn:
            cursor = conn.execute(f"SELECT {col} FROM metrics WHERE vm_id = ? ORDER BY id DESC LIMIT ?", (vm_id, window_size))
            return [row[0] for row in cursor.fetchall()]

class ObservabilitySpoke:
    def __init__(self, config: Config):
        self.config = config
        self.history = {m: {vm: [] for vm in config.VM_LIST} for m in ["rtt", "cpu", "ram"]}
        self.max_points = 50
    def update_dashboard(self, current_data: List[Dict]):
        for entry in current_data:
            vm = entry["vm_id"]
            self.history["rtt"][vm].append(entry.get("rtt_ms", 0))
            self.history["cpu"][vm].append(entry.get("cpu_usage", 0))
            self.history["ram"][vm].append(entry.get("ram_usage", 0))
            for m in ["rtt", "cpu", "ram"]:
                if len(self.history[m][vm]) > self.max_points: self.history[m][vm].pop(0)
    def start_gui(self):
        plt.ion(); fig = plt.figure(figsize=(12, 10))
        axs_rtt = [fig.add_subplot(3, 2, i+1) for i in range(4)]
        ax_avg_cpu = fig.add_subplot(3, 2, 5); ax_avg_ram = fig.add_subplot(3, 2, 6)
        while True:
            for i, vm in enumerate(self.config.VM_LIST):
                axs_rtt[i].clear(); axs_rtt[i].plot(self.history["rtt"][vm]); axs_rtt[i].set_title(f"Latency {vm}"); axs_rtt[i].set_ylim(0, 100)
            ax_avg_cpu.clear(); ax_avg_ram.clear()
            if self.history["cpu"]["vm1"]:
                ax_avg_cpu.plot([statistics.mean([self.history["cpu"][v][t] for v in self.config.VM_LIST]) for t in range(len(self.history["cpu"]["vm1"]))]); ax_avg_cpu.set_title("System Mean CPU (%)")
                ax_avg_ram.plot([statistics.mean([self.history["ram"][v][t] for v in self.config.VM_LIST]) for t in range(len(self.history["ram"]["vm1"]))]); ax_avg_ram.set_title("System Mean RAM (%)")
            plt.tight_layout(); plt.pause(1)

class ConfigSpoke:
    def __init__(self, config: Config): self.config = config

# -----------------------------------------------------------------------------
# HUB (CENTRAL CORE)
# -----------------------------------------------------------------------------

class OrchestratorCore:
    def __init__(self, config: Config):
        self.config = config
        self.db = DatabaseSpoke(config); self.history = HistoryLoaderSpoke(config)
        self.ml_predictor = MLPredictorSpoke(config); self.decision_engine = DecisionIntelligenceSpoke(config)
        self.viz = ObservabilitySpoke(config); self.intent_mgr = IntentManagerSpoke(config)
        self.metrics_mgr = MetricsManagerSpoke(config); self.collector = CollectorSpoke(config)
        self.latency_mgr = LatencyManagerSpoke(config); self.latency_mgr.core = self; self.intent_mgr.core = self
        self.mode = "classic"; self.current_slos = []; self.last_real_data_ts = None; self.start_ts = time.time()
        self.app = Flask(f"{__name__}_core")

    def _setup_routes(self):
        @self.app.route('/latency', methods=['POST'])
        def internal_latency():
            data = request.json
            return jsonify(self.run_enhanced_flow(data) if self.mode == "enhanced" else self.run_classic_flow(data)), 200
        @self.app.route('/slos', methods=['POST'])
        def update_slos():
            self.current_slos = request.json.get("slos", [])
            return jsonify({"status": "updated", "slos": self.current_slos}), 200
        @self.app.route('/mode', methods=['POST'])
        def change_mode():
            self.mode = request.json.get("mode", "classic")
            return jsonify({"status": "changed", "mode": self.mode}), 200
        @self.app.route('/status', methods=['GET'])
        def get_status():
            return jsonify({"mode": self.mode, "current_slos": self.current_slos, "vm_list": self.config.VM_LIST, "uptime_seconds": int(time.time() - self.start_ts)}), 200

    def send_to_master(self, decision: Dict, mode: str) -> bool:
        if decision["decision"] == "stay": return True
        success = self._perform_post(decision, mode)
        if not success: threading.Thread(target=self._retry_loop, args=(decision, mode), daemon=True).start()
        return success

    def _perform_post(self, decision: Dict, mode: str, attempt: int = 1) -> bool:
        try:
            payload = {"decision": "migrate", "service": "my_service", "from_vm": decision["from_vm"], "to_vm": decision["to_vm"], "mode": mode, "timestamp": datetime.now(timezone.utc).isoformat(), "reason": decision["reason"]}
            headers = {"Content-Type": "application/json", "Authorization": f"Bearer {self.config.MASTER_TOKEN}"}
            resp = requests.post(self.config.MASTER_URL, json=payload, headers=headers, timeout=self.config.MASTER_TIMEOUT)
            logger.info(f"[Master] Migration delivery status: {resp.status_code}")
            return resp.status_code == 200
        except Exception:
            logger.warning(f"[Master] Injoignable (tentative {attempt}/3)")
            return False

    def _retry_loop(self, decision: Dict, mode: str):
        for i in range(2):
            time.sleep(2)
            if self._perform_post(decision, mode, attempt=i+2): return
        logger.critical("[Master] ══ ÉCHEC LIVRAISON DÉCISION après 3 tentatives ══")

    def set_user_intent(self, text: str):
        self.current_slos = self.intent_mgr.query_intent_engine(text)
        self.mode = "enhanced"; logger.info(f"[Core] Enhanced Mode Enabled. SLOs: {self.current_slos}")

    def run_classic_flow(self, measurements: List[Dict]):
        print(f"\n{Fore.WHITE}{'═'*60}")
        print(f"{Fore.WHITE}  🔄 NOUVEAU CYCLE — {datetime.now().strftime('%H:%M:%S')} | Mode: {self.mode.upper()}")
        print(f"{Fore.WHITE}{'═'*60}")
        
        # 1. STORE METRICS
        log_step("1. STORE METRICS", "Core → Database", measurements)
        self.db.save_metrics(measurements, "classic")
        
        # 2. LOAD HISTORY
        log_step("2. LOAD HISTORY", "Core → HistoryLoader", {
            "mode": "classic",
            "metrics": ["latency"],
            "vms": self.config.VM_LIST,
            "window": "last_50_records"
        })
        
        # 3. ML PREDICTION
        log_step("3. ML PREDICTION", "Core → MLPredictor", {
            "mode": "classic",
            "vms": [
                {
                    "vm_id": m["vm_id"],
                    "current_latency_ms": m["rtt_ms"],
                    "history_latency": self.history.load_history_window(m["vm_id"], "latency", 5)
                }
                for m in measurements
            ]
        })
        preds = {m["vm_id"]: self.ml_predictor.get_prediction(m["rtt_ms"], self.history.load_history_window(m["vm_id"], "latency", 10)) for m in measurements}
        
        # 4. RETOUR PREDICTIONS
        log_step("4. RETOUR PREDICTIONS", "MLPredictor → Core", {
            "mode": "classic",
            "predictions_latency": [
                {"vm_id": vm_id, "predictions": list(preds[vm_id])}
                for vm_id in preds
            ]
        })
        
        # 5. DECISION REQUEST
        log_step("5. DECISION REQUEST", "Core → DecisionIntelligence", {
            "mode": "classic",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "vms": [
                {
                    "vm_id": m["vm_id"],
                    "current_latency_ms": m["rtt_ms"],
                    "predicted_latencies": list(preds.get(m["vm_id"], []))
                }
                for m in measurements
            ]
        })
        dec = self.decision_engine.evaluate_decision(measurements, preds)
        
        # 6. RETOUR DÉCISION
        log_step("6. RETOUR DÉCISION", "DecisionIntelligence → Core", {
            "decision": dec.get("decision"),
            "from_vm": dec.get("from_vm"),
            "to_vm": dec.get("to_vm"),
            "reason": dec.get("reason")
        })

        self.viz.update_dashboard(measurements)
        
        # 7. MIGRATION COMMAND
        log_step("7. MIGRATION COMMAND", "Core → Master", {
            "decision": dec.get("decision"),
            "service": "my_service",
            "from_vm": dec.get("from_vm"),
            "to_vm": dec.get("to_vm"),
            "mode": "classic",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": dec.get("reason")
        })
        ack = self.send_to_master(dec, "classic")
        self.db.save_decision(dec, "classic", ack)
        
        self._log_final_decision(dec)
        return dec

    def run_enhanced_flow(self, rtt_measurements: List[Dict]):
        print(f"\n{Fore.WHITE}{'═'*60}")
        print(f"{Fore.WHITE}  🔄 NOUVEAU CYCLE — {datetime.now().strftime('%H:%M:%S')} | Mode: {self.mode.upper()}")
        print(f"{Fore.WHITE}{'═'*60}")
        
        # 1. Core → MetricsManager (envoyer les SLOs pour analyse)
        log_step("1. ANALYZE SLOs", "Core → MetricsManager", {
            "intent_id": "current",
            "slos": self.current_slos
        })
        needed = self.metrics_mgr.analyze_slos(self.current_slos)
        
        # 2. MetricsManager → Collector (envoyer la liste des métriques à collecter)
        log_step("2. COLLECTION REQUEST", "MetricsManager → Collector", {
            "mode": "enhanced",
            "metrics_needed": needed,
            "vms": self.config.VM_LIST
        })
        enriched = [{**rtt_m, **self.collector.collect_vm_metrics(rtt_m["vm_id"])} for rtt_m in rtt_measurements]

        # 3. Collector → Database (stocker les métriques collectées CPU/RAM)
        log_step("3. STORE METRICS", "Collector → Database", {
            "mode": "enhanced",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "metrics": [
                {
                    "vm_id": e["vm_id"],
                    "cpu_usage": e.get("cpu_usage", 0),
                    "ram_usage": e.get("ram_usage", 0)
                }
                for e in enriched
            ]
        })
        self.db.save_metrics(enriched, "enhanced")
        
        # 4. Core → HistoryLoader → Database (récupérer l'historique complet)
        log_step("4. LOAD HISTORY", "Core → HistoryLoader → Database", {
            "mode": "enhanced",
            "metrics": ["latency", "cpu_usage", "ram_usage"],
            "vms": self.config.VM_LIST,
            "window": "last_50_records"
        })
        
        # 5. Core → MLPredictor (envoyer données courantes + historique)
        log_step("5. ML PREDICTION REQUEST", "Core → MLPredictor", {
            "mode": "enhanced",
            "vms": [
                {
                    "vm_id": e["vm_id"],
                    "current_latency_ms": e.get("rtt_ms", 0),
                    "cpu_usage": e.get("cpu_usage", 0),
                    "ram_usage": e.get("ram_usage", 0),
                    "history": self.history.load_history_window(e["vm_id"], "latency", 3)
                }
                for e in enriched
            ]
        })
        preds_map = {e["vm_id"]: {m: self.ml_predictor.get_enhanced_prediction(
            m,
            e.get("rtt_ms", 0) if m == "latency" else e.get(m, 0),
            self.history.load_history_window(e["vm_id"], m, 10)
        ) for m in ["latency", "cpu_usage", "ram_usage"]} for e in enriched}
        
        # 6. MLPredictor → Core (retour des prédictions)
        log_step("6. RETOUR PREDICTIONS", "MLPredictor → Core", {
            "mode": "enhanced",
            "predictions": [
                {
                    "vm_id": vm_id,
                    "predictions_latency": list(preds_map[vm_id].get("latency", [])),
                    "predictions_cpu": list(preds_map[vm_id].get("cpu_usage", [])),
                    "predictions_ram": list(preds_map[vm_id].get("ram_usage", []))
                }
                for vm_id in preds_map
            ]
        })
        
        # 7. Core → DecisionIntelligence (envoyer données + prédictions + SLOs)
        log_step("7. DECISION REQUEST", "Core → DecisionIntelligence", {
            "mode": "enhanced",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "slos": self.current_slos,
            "vms": [
                {
                    "vm_id": e["vm_id"],
                    "current_latency_ms": e.get("rtt_ms", 0),
                    "current_cpu_usage": e.get("cpu_usage", 0),
                    "current_ram_usage": e.get("ram_usage", 0)
                }
                for e in enriched
            ]
        })
        dec = self.decision_engine.evaluate_enhanced_decision(enriched, preds_map, self.current_slos)
        
        # 8. DecisionIntelligence → Core (retour de la décision)
        log_step("8. RETOUR DÉCISION", "DecisionIntelligence → Core", {
            "decision": dec.get("decision"),
            "from_vm": dec.get("from_vm"),
            "to_vm": dec.get("to_vm"),
            "reason": dec.get("reason")
        })

        self.viz.update_dashboard(enriched)
        
        # 9. Core → Master (envoyer la commande de migration)
        log_step("9. MIGRATION COMMAND", "Core → Master", {
            "decision": dec.get("decision"),
            "service": "my_service",
            "from_vm": dec.get("from_vm"),
            "to_vm": dec.get("to_vm"),
            "mode": "enhanced",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "reason": dec.get("reason")
        })
        ack = self.send_to_master(dec, "enhanced")
        self.db.save_decision(dec, "enhanced", ack)
        
        self._log_final_decision(dec)
        return dec

    def _log_final_decision(self, dec: Dict):
        if dec["decision"] == "migrate":
            msg = f"══ DÉCISION : MIGRATE {dec.get('from_vm')} → {dec.get('to_vm')} | {dec.get('reason')} ══"
            logger.info(f"{Fore.RED}{Style.BRIGHT}{msg}")
        else:
            msg = f"══ DÉCISION : STAY | {dec.get('reason')} ══"
            logger.info(f"{Fore.GREEN}{msg}")

    def start(self):
        self.db.init_db()
        self.latency_mgr.start_api(); self.intent_mgr.start_api(); self._setup_routes()
        
        print(f"""
{Fore.WHITE}{Style.BRIGHT}   ╔══════════════════════════════════════════════════════╗
   ║     {Fore.CYAN}LAAS-CNRS — VM Migration Orchestrator v2.0      {Fore.WHITE}║
   ║     {Fore.YELLOW}Hub-and-Spoke Architecture                      {Fore.WHITE}║
   ╠══════════════════════════════════════════════════════╣
   ║  {Fore.GREEN}Core      : http://localhost:8000                  {Fore.WHITE}║
   ║  {Fore.GREEN}Latency   : http://localhost:8010/rtt              {Fore.WHITE}║
   ║  {Fore.GREEN}Intent    : http://localhost:8014/intent           {Fore.WHITE}║
   ╚══════════════════════════════════════════════════════╝
        """)

        threading.Thread(target=self.app.run, kwargs={'host': '0.0.0.0', 'port': self.config.CORE_PORT, 'threaded': True, 'use_reloader': False}, daemon=True).start()
        threading.Thread(target=self.viz.start_gui, daemon=True).start()
        if self.mode == "enhanced": self.set_user_intent("Je veux que le service soit tres reactif, latence < 20ms et CPU < 70%")
        while True:
            if self.last_real_data_ts is None or (time.time() - self.last_real_data_ts > 30):
                sim_rtt = [{"vm_id": vm, "rtt_ms": random.uniform(5, 100)} for vm in self.config.VM_LIST]
                if self.mode == "enhanced": self.run_enhanced_flow(sim_rtt)
                else: self.run_classic_flow(sim_rtt)
            time.sleep(self.config.COLLECTION_INTERVAL)

if __name__ == "__main__":
    core = OrchestratorCore(Config())
    try: core.start()
    except KeyboardInterrupt: logger.info("Stopping...")
