import pytest
import sqlite3
import time
import os
import threading
import requests
import gc
from orchestrator import (
    Config, DatabaseSpoke, MLPredictorSpoke, 
    DecisionIntelligenceSpoke, IntentManagerSpoke, OrchestratorCore,
    HistoryLoaderSpoke
)

def _remove_db_safely(db_name: str):
    """Force la fermeture de SQLite et tente de supprimer le fichier avec retry."""
    gc.collect()
    time.sleep(0.1)
    for _ in range(3):
        try:
            if os.path.exists(db_name):
                os.remove(db_name)
            break
        except PermissionError:
            time.sleep(0.2)

# -----------------------------------------------------------------------------
# SECTION 1 : TESTS UNITAIRES (sans Flask, sans réseau)
# -----------------------------------------------------------------------------

def test_1_1_config_defaults():
    """Vérification des paramètres par défaut."""
    conf = Config()
    assert conf.CORE_PORT == 8000
    assert conf.LATENCY_PORT == 8010
    assert conf.DB_NAME == "orchestrator.db"
    assert conf.VM_LIST == ["vm1", "vm2", "vm3", "vm4"]

def test_1_2_database_spoke_lifecycle():
    """Vérifie le cycle init -> save -> load avec nettoyage."""
    test_db = "test_unit_db.db"
    _remove_db_safely(test_db)
    
    try:
        conf = Config(DB_NAME=test_db)
        db = DatabaseSpoke(conf)
        db.init_db()
        
        measurements = [{"vm_id": "vm1", "rtt_ms": 42.0}]
        db.save_metrics(measurements, mode="classic")
        
        # Correction : utilise HistoryLoaderSpoke au lieu de DatabaseSpoke
        history_loader = HistoryLoaderSpoke(conf)
        history = history_loader.load_history_window("vm1", "latency", 10)
        assert len(history) > 0
        assert history[0] == 42.0
    finally:
        _remove_db_safely(test_db)

def test_1_3_ml_predictor_fallback():
    """Vérifie le basculement en simulation si l'API est injoignable."""
    conf = Config(ML_RTT_URL="http://localhost:1/unreachable")
    ml = MLPredictorSpoke(conf)
    
    current_val = 20.0
    # On s'attend à ce que l'exception soit catchée et la simulation retournée
    preds = ml.get_prediction(current_val, [18.0, 19.0, 20.0])
    
    assert len(preds) == 5
    assert preds[0] == pytest.approx(21.0, rel=1e-3) # 20 * 1.05

def test_1_4_decision_logic_classic():
    """Vérifie la règle de décision Classic : migration si seuil > 50ms."""
    conf = Config()
    di = DecisionIntelligenceSpoke(conf)
    
    current_data = [
        {"vm_id": "vm1", "rtt_ms": 80.0}, # Over threshold
        {"vm_id": "vm2", "rtt_ms": 20.0},
        {"vm_id": "vm3", "rtt_ms": 15.0}, # Best target
        {"vm_id": "vm4", "rtt_ms": 30.0}
    ]
    preds_map = {vm["vm_id"]: [vm["rtt_ms"]]*5 for vm in current_data}
    
    decision = di.evaluate_decision(current_data, preds_map)
    assert decision["decision"] == "migrate"
    assert decision["from_vm"] == "vm1"
    assert decision["to_vm"] == "vm3"

def test_1_5_intent_parsing():
    """Vérifie l'extraction des SLOs par Regex."""
    mgr = IntentManagerSpoke(Config())
    slos = mgr._parse_slos("latency < 20 and cpu < 70")
    
    assert any(s["metric"] == "latency" and s["threshold"] == 20.0 for s in slos)
    assert any(s["metric"] == "cpu_usage" and s["threshold"] == 70.0 for s in slos)

def test_1_6_intent_fallback():
    """Vérifie les SLOs par défaut si le LLM est injoignable."""
    conf = Config(INTENT_ENGINE_URL="http://localhost:1/unreachable")
    mgr = IntentManagerSpoke(conf)
    
    slos = mgr.query_intent_engine("I want fast response")
    assert len(slos) == 3
    assert any(s["metric"] == "latency" and s["threshold"] == 50.0 for s in slos)

# -----------------------------------------------------------------------------
# SECTION 2 : TESTS D'INTÉGRATION (sans réseau externe)
# -----------------------------------------------------------------------------

def test_2_1_classic_flow_complete():
    """Test du flux Classic complet sans boucle infinie."""
    db_name = "test_classic_flow.db"
    _remove_db_safely(db_name)
    
    try:
        core = OrchestratorCore(Config(DB_NAME=db_name))
        core.db.init_db()
        
        measurements = [{"vm_id": vm, "rtt_ms": 10.0} for vm in core.config.VM_LIST]
        measurements[0]["rtt_ms"] = 90.0 # Force migration
        
        decision = core.run_classic_flow(measurements)
        
        assert decision["decision"] == "migrate"
        assert decision["from_vm"] == "vm1"
        
        with sqlite3.connect(db_name) as conn:
            res = conn.execute("SELECT count(*) FROM decisions").fetchone()
            assert res[0] > 0
    finally:
        _remove_db_safely(db_name)

def test_2_2_enhanced_flow_complete():
    """Test du flux Enhanced complet sans boucle infinie."""
    db_name = "test_enhanced_flow.db"
    _remove_db_safely(db_name)
    
    try:
        core = OrchestratorCore(Config(DB_NAME=db_name))
        core.db.init_db()
        core.set_user_intent("low latency") # Active mode enhanced
        
        measurements = [{"vm_id": vm, "rtt_ms": 20.0} for vm in core.config.VM_LIST]
        decision = core.run_enhanced_flow(measurements)
        
        assert "decision" in decision
        assert core.mode == "enhanced"
    finally:
        _remove_db_safely(db_name)

def test_2_3_simulation_switch_logic():
    """Vérifie la logique temporelle du basculement simulation."""
    core = OrchestratorCore(Config())
    assert core.last_real_data_ts is None
    
    core.last_real_data_ts = time.time()
    assert (time.time() - core.last_real_data_ts) < 30

# -----------------------------------------------------------------------------
# SECTION 3 : TESTS FLASK (endpoints HTTP)
# -----------------------------------------------------------------------------

@pytest.fixture(scope="module")
def api_core():
    """Initialisation manuelle et démarrage des serveurs Flask."""
    db_name = "test_flask_api.db"
    _remove_db_safely(db_name)
    
    try:
        core = OrchestratorCore(Config(DB_NAME=db_name))
        core.db.init_db()
        # Initialisation manuelle demandée
        core.latency_mgr.core = core
        core.intent_mgr.core = core
        
        # Démarrage des APIs spokes
        core.latency_mgr.start_api()
        core.intent_mgr.start_api()
        
        # Démarrage API Core
        core._setup_routes()
        threading.Thread(target=core.app.run, kwargs={
            'host': '0.0.0.0', 'port': 8000, 
            'threaded': True, 'use_reloader': False
        }, daemon=True).start()
        
        time.sleep(2) # Laisse le temps au boot
        yield core
    finally:
        _remove_db_safely(db_name)

def test_3_1_api_rtt_post(api_core):
    """Vérifie l'endpoint /rtt avec payload valide."""
    payload = {
        "measurements": [{"vm_id": vm, "rtt_ms": 10.0} for vm in api_core.config.VM_LIST]
    }
    resp = requests.post("http://localhost:8010/rtt", json=payload)
    assert resp.status_code == 200
    assert "decision" in resp.json()

def test_3_2_api_rtt_invalid(api_core):
    """Vérifie le rejet d'un payload vide."""
    resp = requests.post("http://localhost:8010/rtt", json={})
    assert resp.status_code == 400

def test_3_3_api_core_status(api_core):
    """Vérifie l'endpoint status du Core."""
    resp = requests.get("http://localhost:8000/status")
    assert resp.status_code == 200
    data = resp.json()
    assert "mode" in data
    assert "vm_list" in data
    assert "uptime_seconds" in data
