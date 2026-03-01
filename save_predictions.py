import psycopg2
from psycopg2.extras import Json
from datetime import datetime
import os

PG_CONN = "host=localhost dbname=xray_db user=xray_user password=Chandu#01"
MODEL_VERSION = "efficientnet_v1"

def save_prediction_to_db(image_path, pred_probs, pred_label, gradcam_path=None):

    conn = psycopg2.connect(PG_CONN)
    cur = conn.cursor()

    # 1. Find the image row
    cur.execute("SELECT id FROM xray_images WHERE storage_path=%s LIMIT 1;", (image_path,))
    row = cur.fetchone()

    # 2. If missing, create one
    if row:
        image_id = row[0]
    else:
        cur.execute("""
            INSERT INTO xray_images (filename, storage_path, uploaded_by, dataset_split)
            VALUES (%s,%s,%s,%s)
            RETURNING id;
        """, (os.path.basename(image_path), image_path, "inference", "test"))
        image_id = cur.fetchone()[0]

    # 3. Insert prediction
    cur.execute("""
        INSERT INTO xray_predictions (
            image_id, model_version, predicted_label,
            predicted_probs, confidence, gradcam_path,
            run_at, run_by
        )
        VALUES (%s,%s,%s,%s,%s,%s,%s,%s)
    """, (
        image_id,
        MODEL_VERSION,
        pred_label,
        Json(pred_probs),
        max(pred_probs.values()),
        gradcam_path,
        datetime.utcnow(),
        "windows_inference"
    ))

    conn.commit()
    cur.close()
    conn.close()

    return True
