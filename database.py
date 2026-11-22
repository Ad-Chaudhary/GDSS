import re
import sqlite3
import datetime
import numpy as np
import pandas as pd

DB_FILE = "groundwater_system.db"


def get_db_connection(timeout=30.0):
    conn = sqlite3.connect(DB_FILE, timeout=timeout)
    try:
        conn.execute("PRAGMA busy_timeout=30000;")
    except Exception:
        pass
    return conn


def evaluate_telemetry_status(extraction_m3, recharge_m3, predicted_depth_mbgl=10.0):
    if recharge_m3 is not None and float(recharge_m3) > 0:
        soe = (float(extraction_m3) / float(recharge_m3)) * 100.0
    else:
        soe = 0.0

    if soe <= 70.0:
        assigned_status = "SAFE"
        color_code = "#22c55e"
    elif 70.0 < soe <= 90.0:
        assigned_status = "SEMI_CRITICAL"
        color_code = "#eab308"
    elif 90.0 < soe <= 100.0:
        assigned_status = "CRITICAL"
        color_code = "#f97316"
    else:
        assigned_status = "OVER_EXPLOITED"
        color_code = "#ef4444"

    if predicted_depth_mbgl is not None and not (isinstance(predicted_depth_mbgl, float) and np.isnan(predicted_depth_mbgl)):
        bounded_depth = float(np.clip(predicted_depth_mbgl, a_min=1.0, a_max=150.0))
    else:
        bounded_depth = 10.0

    return {
        "soe_percent": round(soe, 2),
        "status": assigned_status,
        "color": color_code,
        "predicted_depth": round(bounded_depth, 2),
        "predicted_depth_mbgl": round(bounded_depth, 2)
    }


def classify_status(soe=None, depth=None):
    if soe is not None and not (isinstance(soe, float) and np.isnan(soe)):
        soe_val = float(soe)
        if soe_val <= 70.0:
            return "SAFE"
        elif soe_val <= 90.0:
            return "SEMI_CRITICAL"
        elif soe_val <= 100.0:
            return "CRITICAL"
        else:
            return "OVER_EXPLOITED"

    if depth is not None and not (isinstance(depth, float) and np.isnan(depth)):
        depth_val = float(np.clip(depth, 1.0, 150.0))
        if depth_val < 6.0:
            return "SAFE"
        elif depth_val < 12.0:
            return "SEMI_CRITICAL"
        elif depth_val <= 22.0:
            return "CRITICAL"
        else:
            return "OVER_EXPLOITED"

    return "SAFE"


def evaluate_cgwb_status(soe):
    soe_val = float(soe) if soe is not None else 0.0
    if soe_val <= 70.0:
        status = "SAFE"
        color = "#22c55e"
    elif soe_val <= 90.0:
        status = "SEMI_CRITICAL"
        color = "#eab308"
    elif soe_val <= 100.0:
        status = "CRITICAL"
        color = "#f97316"
    else:
        status = "OVER_EXPLOITED"
        color = "#ef4444"
    return status, color, f"SOE: {round(soe_val, 2)}%", f"CGWB Status: {status}"


def classify_depth(depth, soe=None):
    return classify_status(soe=soe, depth=depth)


def recalculate_db_statuses():
    conn = get_db_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT id, extraction_m3, recharge_m3, depth_m FROM telemetry")
    rows = cursor.fetchall()
    if rows:
        updates = []
        for r_id, ext, rec, depth in rows:
            res = evaluate_telemetry_status(ext, rec, depth)
            updates.append((res["soe_percent"], res["status"], r_id))
        cursor.executemany("UPDATE telemetry SET soe_percentage = ?, status = ? WHERE id = ?", updates)
        conn.commit()
    conn.close()


def init_db():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stations (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_code TEXT UNIQUE,
            name TEXT,
            state TEXT,
            district TEXT,
            aquifer_type TEXT,
            latitude REAL,
            longitude REAL,
            specific_yield REAL
        )
    """)

    cursor.execute("PRAGMA table_info(stations)")
    columns = [column[1] for column in cursor.fetchall()]
    if "aquifer_type" not in columns:
        cursor.execute("ALTER TABLE stations ADD COLUMN aquifer_type TEXT DEFAULT 'Alluvium'")

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_code TEXT,
            timestamp DATETIME,
            rainfall_mm REAL,
            depth_m REAL,
            extraction_m3 REAL,
            recharge_m3 REAL,
            soe_percentage REAL,
            status TEXT,
            FOREIGN KEY (station_code) REFERENCES stations (station_code)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS predictions (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            station_code TEXT,
            timestamp DATETIME,
            rainfall_mm REAL,
            extraction_m3 REAL,
            recharge_m3 REAL,
            soe_percentage REAL,
            predicted_depth_m REAL,
            predicted_status TEXT,
            confidence REAL,
            model_r2 REAL,
            model_accuracy REAL,
            FOREIGN KEY (station_code) REFERENCES stations (station_code)
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS officers (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            designation TEXT UNIQUE,
            email TEXT
        )
    """)

    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_station_ts ON telemetry(station_code, timestamp);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_telemetry_ts ON telemetry(timestamp);")

    conn.commit()
    conn.close()


def seed_fallback_data():
    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT COUNT(*) FROM stations")
    if cursor.fetchone()[0] == 0:
        demo_stations = [
            ("DWLR_UP_0070", "Jirauli 1", "Uttar Pradesh", "Buland Shahar", "Alluvium", 28.40, 77.85, 0.15),
            ("DWLR_UP_0155", "Tejkishan Khera_1", "Uttar Pradesh", "Lucknow", "Alluvium", 26.84, 80.94, 0.15),
            ("DWLR_UP_0140", "Sapti Nankar_1", "Uttar Pradesh", "Siddharthnagar", "Alluvium", 27.27, 82.82, 0.15),
            ("DWLR_UP_0012", "Bansepur", "Uttar Pradesh", "Azamgarh", "Alluvium", 26.05, 83.18, 0.15),
            ("DWLR_UP_0051", "Fugana_2", "Uttar Pradesh", "Muzaffarnagar", "Alluvium", 29.47, 77.70, 0.15),
            ("DWLR_IN_4463", "Bihar DWLR-100 (Purnia)", "Bihar", "Purnia", "Gangetic Alluvium", 25.77, 87.47, 0.12),
            ("DWLR_IN_4560", "Bihar DWLR-197 (Gaya)", "Bihar", "Gaya", "Gangetic Alluvium", 24.79, 85.00, 0.12),
            ("DWLR_IN_4439", "Bihar DWLR-076 (Patna)", "Bihar", "Patna", "Gangetic Alluvium", 25.59, 85.13, 0.12),
            ("AS-KAM-001", "Kamrup Metro-1", "Assam", "Kamrup Metropolitan", "Alluvial", 26.15, 91.77, 0.15),
            ("AS-JOR-002", "Jorhat-2", "Assam", "Jorhat", "Alluvial", 26.75, 94.22, 0.12)
        ]
        cursor.executemany("""
            INSERT INTO stations (station_code, name, state, district, aquifer_type, latitude, longitude, specific_yield)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, demo_stations)

        np.random.seed(42)
        records = []
        base_time = datetime.datetime.now() - datetime.timedelta(days=90)

        station_codes = [s[0] for s in demo_stations]
        for st_code in station_codes:
            for i in range(90):
                ts = (base_time + datetime.timedelta(days=i)).strftime("%Y-%m-%d %H:%M:%S")
                rf = max(0.0, float(np.random.uniform(0.0, 250.0)))
                ext = float(np.random.uniform(10.0, 1000.0))
                rec = float(np.random.uniform(100.0, 900.0))
                depth_val = 8.6 + (ext / (rec + 10.0)) * 10.8 - (rf / 38.0) + np.random.normal(0, 0.2)
                depth_val = float(np.clip(depth_val, 2.0, 120.0))
                eval_res = evaluate_telemetry_status(ext, rec, depth_val)
                records.append((st_code, ts, round(rf, 2), round(depth_val, 2), round(ext, 2), round(rec, 2), eval_res["soe_percent"], eval_res["status"]))

        cursor.executemany("""
            INSERT INTO telemetry (station_code, timestamp, rainfall_mm, depth_m, extraction_m3, recharge_m3, soe_percentage, status)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, records)

    cursor.execute("SELECT COUNT(*) FROM officers")
    if cursor.fetchone()[0] == 0:
        demo_officers = [
            ("Dr. Rajesh Sharma", "District Collector & Magistrate", "collector.office@gov.in"),
            ("Er. Ananya Gogoi", "CGWB Nodal Regional Officer", "cgwb.nodal@gov.in"),
            ("Pritam Baruah", "Chief Hydrogeologist", "hydro.chief@gov.in"),
            ("Sanjay Sen", "Water Resource Department Engineer", "wrd.engineer@gov.in")
        ]
        cursor.executemany("""
            INSERT INTO officers (name, designation, email)
            VALUES (?, ?, ?)
        """, demo_officers)

    conn.commit()
    conn.close()


def get_officers():
    conn = get_db_connection()
    df = pd.read_sql_query("SELECT id, name, designation, email FROM officers ORDER BY name", conn)
    conn.close()
    return df


def add_officer(name, designation, email):
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO officers (name, designation, email) VALUES (?, ?, ?)
        ON CONFLICT(designation) DO UPDATE SET name=excluded.name, email=excluded.email
    """, (name.strip(), designation.strip(), email.strip()))
    conn.commit()
    conn.close()


def delete_officer(officer_id):
    conn = get_db_connection()
    conn.execute("DELETE FROM officers WHERE id = ?", (officer_id,))
    conn.commit()
    conn.close()


def is_valid_email(email):
    return bool(re.match(r"^[^@\s]+@[^@\s]+\.[^@\s]+$", email or ""))


def save_prediction(station, ts, rf, ext, rec, soe, depth, status, confidence, r2, accuracy):
    bounded_depth = float(np.clip(depth, a_min=1.0, a_max=150.0))
    conn = get_db_connection()
    conn.execute("""
        INSERT INTO predictions (station_code, timestamp, rainfall_mm, extraction_m3, recharge_m3,
                                 soe_percentage, predicted_depth_m, predicted_status, confidence,
                                 model_r2, model_accuracy)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (station, ts, rf, ext, rec, soe, bounded_depth, status, confidence, r2, accuracy))
    conn.commit()
    conn.close()