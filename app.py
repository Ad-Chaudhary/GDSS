import datetime
import sqlite3
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

import importlib
import database
import ml_models
importlib.reload(database)
importlib.reload(ml_models)

from database import (
    DB_FILE,
    get_db_connection,
    add_officer,
    evaluate_telemetry_status,
    evaluate_cgwb_status,
    classify_status,
    classify_depth,
    delete_officer,
    get_officers,
    init_db,
    is_valid_email,
    save_prediction,
    seed_fallback_data,
)
from ml_models import (
    CATEGORY_DETAILS,
    MIN_TRAIN_ROWS,
    FEATURES,
    build_report_text,
    clean_telemetry_df,
    fetch_data_gov_json_api,
    get_prediction_features,
    predict_water_table_depth,
    predict_risk_proba,
    process_and_ingest_dataframe,
    render_model_performance,
    report_table_for_display,
    send_email_report,
    train_models,
)

st.set_page_config(page_title="Groundwater Intelligence System", layout="wide")

init_db()
seed_fallback_data()
models = train_models()

st.title("GROUNDWATER DECISION SUPPORT SYSTEM")
st.markdown("**National Water Informatics Centre (NWIC) & CGWB Telemetry Network**")

st.sidebar.header("Filter & Navigation")

conn = get_db_connection()
stations_df = pd.read_sql_query("SELECT * FROM stations", conn)
conn.close()

states_list = ["All States"] + sorted(list(stations_df["state"].dropna().unique()))
selected_state = st.sidebar.selectbox("Select State", states_list)

filtered_stations = (
    stations_df
    if selected_state == "All States"
    else stations_df[stations_df["state"] == selected_state]
)

names_list = ["All Stations"] + sorted(list(filtered_stations["name"].dropna().unique()))
selected_name = st.sidebar.selectbox("Select Station Name", names_list)

if selected_name != "All Stations":
    filtered_stations = filtered_stations[filtered_stations["name"] == selected_name]

ids_list = sorted(list(filtered_stations["station_code"].dropna().unique()))
selected_station_id = st.sidebar.selectbox(
    "Select Station ID", ids_list if ids_list else stations_df["station_code"].tolist()
)

conn = get_db_connection()
if selected_state == "All States":
    latest_telemetry = pd.read_sql_query("""
        SELECT t.*, s.name, s.state, s.district, s.aquifer_type 
        FROM telemetry t 
        JOIN stations s ON t.station_code = s.station_code
        WHERE t.id IN (SELECT MAX(id) FROM telemetry GROUP BY station_code)
    """, conn)
else:
    latest_telemetry = pd.read_sql_query("""
        SELECT t.*, s.name, s.state, s.district, s.aquifer_type 
        FROM telemetry t 
        JOIN stations s ON t.station_code = s.station_code
        WHERE s.state = ? AND t.id IN (SELECT MAX(id) FROM telemetry GROUP BY station_code)
    """, conn, params=(selected_state,))

selected_station_telemetry = pd.read_sql_query(
    "SELECT * FROM telemetry WHERE station_code = ? ORDER BY timestamp ASC",
    conn, params=(selected_station_id,)
)
selected_station_predictions = pd.read_sql_query(
    "SELECT timestamp, predicted_depth_m FROM predictions WHERE station_code = ? ORDER BY timestamp ASC",
    conn, params=(selected_station_id,)
)
conn.close()

tab_exec, tab_ingest, tab_sync = st.tabs([
    "Overview & Executive Report", 
    "Prediction of Water Table & Risk Status", 
    "Telemetry Data Ingestion & Cleaning"
])

with tab_exec:
    st.caption(f"Target Region: **{selected_state}** | Generated: **{datetime.date.today().strftime('%Y-%m-%d')}**")
    
    st.subheader("1. Executive Overview & Key Hydrological Metrics (CGWB 2026 Guidelines)")
    total_sensors = len(latest_telemetry)
    avg_depth = round(latest_telemetry["depth_m"].mean(), 2) if total_sensors > 0 else 0.0
    
    safe_cnt = len(latest_telemetry[latest_telemetry["status"] == "SAFE"])
    semi_critical_cnt = len(latest_telemetry[latest_telemetry["status"] == "SEMI_CRITICAL"])
    critical_cnt = len(latest_telemetry[latest_telemetry["status"] == "CRITICAL"])
    overexp_cnt = len(latest_telemetry[latest_telemetry["status"] == "OVER_EXPLOITED"])
    
    safe_pct = round((safe_cnt / total_sensors) * 100, 1) if total_sensors > 0 else 0.0
    semi_critical_pct = round((semi_critical_cnt / total_sensors) * 100, 1) if total_sensors > 0 else 0.0
    critical_pct = round((critical_cnt / total_sensors) * 100, 1) if total_sensors > 0 else 0.0
    overexp_pct = round((overexp_cnt / total_sensors) * 100, 1) if total_sensors > 0 else 0.0
    
    metrics_summary_df = pd.DataFrame([
        {"Metric Parameter": "Total DWLR Sensors Ingesting Telemetry", "Observed Value": f"{total_sensors}", "Status / Standard": "Active Network"},
        {"Metric Parameter": "Average Water Table Depth", "Observed Value": f"{avg_depth} m MBGL", "Status / Standard": "Meters Below Ground Level (1-150m)"},
        {"Metric Parameter": "Safe Category Sensors (SOE ≤ 70%)", "Observed Value": f"{safe_cnt} ({safe_pct}%)", "Status / Standard": "Low Risk / Sustainable"},
        {"Metric Parameter": "Semi-Critical Category Sensors (70% < SOE ≤ 90%)", "Observed Value": f"{semi_critical_cnt} ({semi_critical_pct}%)", "Status / Standard": "Moderate Risk"},
        {"Metric Parameter": "Critical Category Sensors (90% < SOE ≤ 100%)", "Observed Value": f"{critical_cnt} ({critical_pct}%)", "Status / Standard": "High Depletion Risk"},
        {"Metric Parameter": "Over-Exploited Category Sensors (SOE > 100%)", "Observed Value": f"{overexp_cnt} ({overexp_pct}%)", "Status / Standard": "Severe Over-Extraction"},
    ])
    st.table(metrics_summary_df)
    st.markdown("---")
    
    st.subheader("2. Top High-Risk Depletion Hotspots (Immediate Action Required)")
    if not latest_telemetry.empty:
        hotspots_df = latest_telemetry.sort_values(by="depth_m", ascending=False).head(8)
        hotspots_display = hotspots_df[["station_code", "name", "district", "state", "aquifer_type", "depth_m", "soe_percentage", "status"]].copy()
        hotspots_display.columns = ["Station ID", "Station Name", "District", "State", "Aquifer", "Water Depth (MBGL)", "Stage of Extraction (SOE %)", "CGWB Risk Category"]
        hotspots_display["Water Depth (MBGL)"] = hotspots_display["Water Depth (MBGL)"].astype(str) + " m"
        hotspots_display["Stage of Extraction (SOE %)"] = hotspots_display["Stage of Extraction (SOE %)"].astype(str) + " %"
        st.dataframe(hotspots_display, use_container_width=True, hide_index=True)
    else:
        st.info("No telemetry logs available for current filter selection.")
        
    st.markdown("---")

    st.subheader(f"3. Station Telemetry Hydrograph ({selected_station_id})")
    if not selected_station_telemetry.empty:
        c1, c2 = st.columns(2)
        with c1:
            fig_depth = px.line(
                selected_station_telemetry, 
                x="timestamp", 
                y="depth_m", 
                labels={"timestamp": "Date / Time", "depth_m": "Water Level Depth (MBGL)"},
                title="Water Depth Trend (MBGL) - measured vs model predictions",
                markers=True
            )
            if not selected_station_predictions.empty:
                fig_depth.add_trace(go.Scatter(
                    x=selected_station_predictions["timestamp"],
                    y=selected_station_predictions["predicted_depth_m"],
                    mode="markers", name="Model prediction",
                    marker=dict(symbol="x", size=11, color="red"),
                ))
            fig_depth.update_yaxes(autorange="reversed")
            st.plotly_chart(fig_depth, use_container_width=True)
            
        with c2:
            fig_bal = go.Figure()
            fig_bal.add_trace(go.Scatter(x=selected_station_telemetry["timestamp"], y=selected_station_telemetry["extraction_m3"], name="Extraction (m³)"))
            fig_bal.add_trace(go.Scatter(x=selected_station_telemetry["timestamp"], y=selected_station_telemetry["recharge_m3"], name="Recharge (m³)"))
            fig_bal.update_layout(title="Volumetric Water Balance", xaxis_title="Date / Time", yaxis_title="Volume (m³)")
            st.plotly_chart(fig_bal, use_container_width=True)

    st.markdown("---")
    st.subheader("4. Policy & Artificial Recharge Recommendations")
    st.markdown("""
    * **Priority Artificial Recharge:** Construct check dams and percolation tanks in Over-Exploited (SOE > 100%) and Critical (90–100%) blocks.
    * **Micro-Irrigation Mandate:** Shift agricultural abstraction to drip and sprinkler systems to reduce draft by 25%.
    * **Telemetry Verification:** Recalibrate DWLR sensors exhibiting Kalman Filter outlier flags exceeding 15%.
    """)

with tab_ingest:
    st.subheader("Telemetry Assessment & Impact Analysis")
    st.markdown("Enter physical telemetry parameters to predict the water table depth and CGWB risk category.")
    st.caption("Predictions are saved in a separate prediction log and are **never** used as training data.")
    
    with st.form("ingest_form"):
        col_in1, col_in2 = st.columns(2)
        with col_in1:
            st_code = st.selectbox("Station Code", stations_df["station_code"].tolist())
            rf = st.number_input("Precipitation / Rainfall (mm)", min_value=0.0, max_value=500.0, value=25.0)
        with col_in2:
            ext = st.number_input("Groundwater Extraction (m³)", min_value=0.0, max_value=5000.0, value=300.0)
            rec = st.number_input("Estimated Recharge (m³)", min_value=1.0, max_value=5000.0, value=350.0)
        
        submit = st.form_submit_button("Run Prediction")

    if submit:
        if models is None:
            st.error(f"Not enough measured history to train the models yet "
                     f"(need at least {MIN_TRAIN_ROWS} usable readings). Import more data first.")
        else:
            feature_df, n_hist = get_prediction_features(st_code, rf, ext, rec)
            raw_input_vec = feature_df.iloc[0].values

            reg_model = models["reg"]

            predicted_depth = predict_water_table_depth(
                model=reg_model,
                feature_scaler=models.get("feature_scaler"),
                target_scaler=models.get("target_scaler"),
                raw_input_vector=feature_df
            )

            eval_res = evaluate_telemetry_status(extraction_m3=ext, recharge_m3=rec, predicted_depth_mbgl=predicted_depth)
            soe = eval_res["soe_percent"]
            predicted_status = eval_res["status"]
            status_color = eval_res["color"]

            proba = predict_risk_proba(models["clf"], feature_df)[0]
            best = int(np.argmax(proba))
            confidence = round(float(proba[best]) * 100.0, 1)
            ts_now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

            m = models["metrics"]
            save_prediction(
                st_code, ts_now, rf, ext, rec, soe, predicted_depth, predicted_status, confidence,
                m["r2"] if m else None, m["accuracy"] if m else None,
            )

            st.session_state["latest_assessment"] = {
                "station": st_code,
                "timestamp": ts_now,
                "rainfall": rf,
                "extraction": ext,
                "recharge": rec,
                "soe": soe,
                "predicted_depth": predicted_depth,
                "status": predicted_status,
                "color": status_color,
                "confidence": confidence,
                "history_readings": n_hist,
            }
            st.success("Prediction generated and saved to the prediction log.")

    if "latest_assessment" in st.session_state:
        assessment = st.session_state["latest_assessment"]
        predicted_status = assessment["status"]
        cat_info = CATEGORY_DETAILS.get(predicted_status, CATEGORY_DETAILS["SAFE"])
        metrics_now = models["metrics"] if models else None

        st.markdown("---")
        st.markdown("### Prediction Summary & CGWB Environmental Criteria")

        res_col1, res_col2, res_col3 = st.columns(3)
        res_col1.metric("Predicted Depth to Water Table", f"{assessment['predicted_depth']} m MBGL")
        res_col2.metric("Calculated Stage of Extraction (SOE)", f"{assessment['soe']}%")
        res_col3.metric("Prediction Confidence", f"{assessment['confidence']}%")

        if metrics_now:
            st.caption(f"Typical depth error on unseen data: about ±{metrics_now['mae']:.2f} m (hold-out MAE).")
        if assessment["history_readings"] < 7:
            st.warning(f"Only {assessment['history_readings']} measured reading(s) exist for this station, "
                       "so history-based features are approximated. Treat this prediction with extra caution.")

        badge_str = cat_info.get("badge", f"{predicted_status}")
        st.markdown(f"#### Assigned Status: {badge_str}")
        
        st.info(f"""
        **CGWB Assessment Criteria Details:**
        * **Stage of Extraction (SOE) Criterion:** {cat_info['soe']}
        * **Hydrological Impact Trend:** {cat_info['trend']}
        * **Recommended Action:** {cat_info['action']}
        """)

        st.markdown("---")
        render_model_performance(models)

        st.markdown("---")
        st.markdown("Report, Download & Notify Authority")

        _meta = stations_df[stations_df["station_code"] == assessment["station"]]
        station_row = _meta.iloc[0].to_dict() if not _meta.empty else {}

        report_text = build_report_text(assessment, station_row, cat_info, metrics_now)

        st.download_button(
            label="Download Prediction Report",
            data=report_text,
            file_name=f"groundwater_report_{assessment['station']}_{datetime.date.today()}.txt",
            mime="text/plain",
        )

        officers_live = get_officers()

        if officers_live.empty:
            st.warning("No officers in the database yet. Add one below.")
        else:
            officer_labels = [
                f"{r['name']} — {r['designation']} ({r['email']})"
                for _, r in officers_live.iterrows()
            ]
            chosen_label = st.selectbox("Select Authority to Notify", officer_labels)
            chosen_officer = officers_live.iloc[officer_labels.index(chosen_label)]

            default_subject = f"Groundwater Alert: {assessment['station']} - {assessment['status']}"
            default_body = (
                f"Respected {chosen_officer['name']},\n\n"
                f"Please find attached the groundwater prediction report for station "
                f"{assessment['station']}.\n\n"
                f"Predicted depth: {assessment['predicted_depth']} m MBGL\n"
                f"Stage of Extraction: {assessment['soe']}%\n"
                f"Status: {assessment['status']} (confidence {assessment['confidence']}%)\n"
                f"Recommended action: {cat_info['action']}\n\n"
                f"Regards,\nGroundwater Intelligence System"
            )

            mail_subject = st.text_input("Subject", value=default_subject)
            mail_body = st.text_area("Message", value=default_body, height=220)

            with st.expander("Sender email settings (Gmail)", expanded=False):
                sender_email = st.text_input("Your Gmail address")
                sender_password = st.text_input(
                    "Gmail App Password", type="password",
                    help="Google Account > Security > 2-Step Verification > App passwords"
                )

            if st.button("Send Report to Authority"):
                if not is_valid_email(sender_email) or not sender_password:
                    st.warning("Please enter your Gmail address and App Password in 'Sender email settings'.")
                elif not is_valid_email(chosen_officer["email"]):
                    st.error("Selected officer has an invalid email address.")
                else:
                    with st.spinner("Sending email..."):
                        ok, info = send_email_report(
                            sender_email, sender_password, chosen_officer["email"],
                            mail_subject, mail_body,
                            attachment_text=report_text,
                            attachment_name=f"groundwater_report_{assessment['station']}.txt",
                        )
                    (st.success if ok else st.error)(info)

    with st.expander("Manage Authority Directory (add / remove officers)"):
        with st.form("add_officer_form", clear_on_submit=True):
            o1, o2, o3 = st.columns(3)
            o_name = o1.text_input("Name")
            o_desig = o2.text_input("Designation")
            o_mail = o3.text_input("Email")
            if st.form_submit_button("Save Officer"):
                if not o_name or not o_desig or not is_valid_email(o_mail):
                    st.warning("Enter a name, designation and a valid email.")
                else:
                    add_officer(o_name, o_desig, o_mail)
                    st.success(f"Saved {o_name}.")
                    st.rerun()

        current = get_officers()
        st.dataframe(current.drop(columns=["id"]), use_container_width=True, hide_index=True)

        if not current.empty:
            del_labels = [f"{r['name']} — {r['designation']}" for _, r in current.iterrows()]
            del_label = st.selectbox("Remove officer", del_labels, key="del_officer_select")
            if st.button("Delete Selected Officer"):
                delete_officer(int(current.iloc[del_labels.index(del_label)]["id"]))
                st.rerun()

    with st.expander("Prediction Log (kept separate from training data)"):
        conn = get_db_connection()
        pred_log = pd.read_sql_query("""
            SELECT timestamp, station_code, predicted_depth_m, predicted_status, confidence,
                   soe_percentage, model_r2, model_accuracy
            FROM predictions ORDER BY id DESC LIMIT 25
        """, conn)
        conn.close()
        if pred_log.empty:
            st.caption("No predictions recorded yet.")
        else:
            st.dataframe(pred_log, use_container_width=True, hide_index=True)

with tab_sync:
    st.subheader("External Data Integration & File Sync Hub")
    st.markdown("Upload offline data files side-by-side with official government API streams. "
                "All incoming data is **cleaned and preprocessed**, then the ML models are retrained and re-evaluated.")
    
    col_upload, col_api = st.columns(2, gap="large")
    
    with col_upload:
        st.subheader("Upload CSV or Excel File")
        st.caption("Upload offline telemetry datasets (.csv, .xlsx, .xls)")
        
        uploaded_file = st.file_uploader(
            "Choose a CSV or Excel file", 
            type=["csv", "xlsx", "xls"],
            key="tab3_file_uploader"
        )
        
        if uploaded_file is not None:
            try:
                if uploaded_file.name.endswith(".csv"):
                    df_upload = pd.read_csv(uploaded_file)
                else:
                    df_upload = pd.read_excel(uploaded_file)
                
                st.write("##### Raw Data Preview")
                st.dataframe(df_upload.head(5), use_container_width=True)

                preview_clean, preview_report = clean_telemetry_df(df_upload)
                with st.expander("Cleaning preview (what will happen to this file)", expanded=True):
                    report_table_for_display(preview_report, include_db=False)
                    if not preview_clean.empty:
                        st.write("Cleaned data preview")
                        st.dataframe(preview_clean.head(5), use_container_width=True)
                
                if st.button("Confirm & Ingest Uploaded File"):
                    with st.spinner("Cleaning records, saving and retraining models..."):
                        inserted, rep = process_and_ingest_dataframe(df_upload)
                        st.session_state["last_ingest"] = {
                            "source": uploaded_file.name, "inserted": inserted, "report": rep}
                        st.cache_resource.clear()
                        st.rerun()
            except Exception as e:
                st.error(f"Error reading file: {str(e)}")

    with col_api:
        st.subheader("Data.gov.in API Key Interface")
        st.caption("Fetch real-time feeds directly from government hydrometric endpoints")
        
        api_key_input = st.text_input("Data.gov.in API Key", type="password", help="Enter 32-character API Key")
        resource_id_input = st.text_input("Resource Identifier", value="579b464d-b66e-4c2b-bdd0-000001c...")
        record_limit = st.slider("Max Records to Pull", min_value=10, max_value=500, value=100)
        
        if st.button("Fetch & Sync API Telemetry"):
            if not api_key_input:
                st.warning("Please provide a valid Data.gov.in API Key.")
            else:
                with st.spinner("Connecting to REST endpoint..."):
                    success, msg, api_report = fetch_data_gov_json_api(api_key_input, resource_id_input, record_limit)
                    if success:
                        st.session_state["last_ingest"] = {
                            "source": "Data.gov.in API",
                            "inserted": api_report["rows_inserted"],
                            "report": api_report,
                        }
                        st.cache_resource.clear()
                        st.rerun()
                    else:
                        st.error(msg)

    if "last_ingest" in st.session_state:
        li = st.session_state["last_ingest"]
        st.success(f"Last import: **{li['source']}** - {li['inserted']} new clean records saved. Models retrained.")
        with st.expander("Data cleaning & preprocessing report (last import)", expanded=True):
            report_table_for_display(li["report"], include_db=True)

    st.markdown("---")

    with st.expander("Need to Add an Individual Record Manually?", expanded=False):
        st.markdown("Fill in parameters for a single monitoring station to add it directly into the system.")
        
        with st.form("manual_entry_form"):
            m_col1, m_col2, m_col3 = st.columns(3)
            with m_col1:
                m_code = st.text_input("Station Code*", value="DWLR_MANUAL_101")
                m_name = st.text_input("Station Name*", value="Custom DWLR Site")
                m_state = st.text_input("State*", value="Uttar Pradesh")
                m_district = st.text_input("District*", value="Ghaziabad")
            with m_col2:
                m_aquifer = st.selectbox("Aquifer Type", ["Alluvium", "Gangetic Alluvium", "Hard Rock", "Basalt", "Sandstone"])
                m_lat = st.number_input("Latitude", value=28.66, format="%.4f")
                m_lon = st.number_input("Longitude", value=77.43, format="%.4f")
                m_sy = st.number_input("Specific Yield", value=0.15, format="%.3f")
            with m_col3:
                m_depth = st.number_input("Water Depth (MBGL)*", min_value=1.0, max_value=150.0, value=12.5)
                m_rf = st.number_input("Rainfall (mm)", min_value=0.0, max_value=1000.0, value=15.0)
                m_ext = st.number_input("Groundwater Extraction (m³)", min_value=0.0, max_value=5000.0, value=350.0)
                m_rec = st.number_input("Estimated Recharge (m³)", min_value=1.0, max_value=5000.0, value=300.0)
            
            manual_submit = st.form_submit_button("Save Entry")
            
            if manual_submit:
                if not m_code or not m_name:
                    st.warning("Please fill in required fields (Station Code and Station Name).")
                else:
                    manual_df = pd.DataFrame([{
                        "station_code": m_code,
                        "station_name": m_name,
                        "state": m_state,
                        "district": m_district,
                        "aquifer_type": m_aquifer,
                        "latitude": m_lat,
                        "longitude": m_lon,
                        "specific_yield": m_sy,
                        "depth_m": m_depth,
                        "rainfall_mm": m_rf,
                        "extraction_m3": m_ext,
                        "recharge_m3": m_rec,
                        "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    }])
                    inserted, rep = process_and_ingest_dataframe(manual_df)
                    st.session_state["last_ingest"] = {"source": f"Manual entry ({m_code})", "inserted": inserted, "report": rep}
                    st.cache_resource.clear()
                    st.rerun()