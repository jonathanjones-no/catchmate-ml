"""Database connection and data fetching for training/prediction."""

import base64
import json
import tempfile
import os
import psycopg2
import psycopg2.extras
from .config import settings


def get_connection():
    return psycopg2.connect(settings.database_url.get_secret_value())


def fetch_training_data():
    """Pull swipe, match, session, and user data for model training."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # All swipes (the core training signal)
            cur.execute("""
                SELECT swiper_id, swiped_id, direction,
                       EXTRACT(EPOCH FROM created_at) as ts
                FROM public.swipes
                ORDER BY created_at
            """)
            swipes = cur.fetchall()

            # User features — join pre-computed intrinsic scores
            cur.execute("""
                SELECT u.id, u.sport_types, u.age, u.is_verified,
                       u.photo_url IS NOT NULL as has_photo,
                       u.bio IS NOT NULL AND u.bio != '' as has_bio,
                       ST_Y(u.location::geometry) as lat,
                       ST_X(u.location::geometry) as lng,
                       EXTRACT(EPOCH FROM (now() - u.created_at)) / 86400 as days_old,
                       EXTRACT(EPOCH FROM (now() - u.last_active_at)) / 86400 as days_inactive,
                       COALESCE(uis.intrinsic_base, 0) as intrinsic_base,
                       COALESCE(uis.initiative, 0) as initiative,
                       COALESCE(uis.total_interactions, 0) as total_interactions
                FROM public.users u
                LEFT JOIN public.user_intrinsic_scores uis ON uis.user_id = u.id
                WHERE u.first_name IS NOT NULL AND u.first_name != ''
                  AND u.is_test_account = false
            """)
            users = cur.fetchall()

            # Match outcomes (positive labels)
            cur.execute("""
                SELECT user1_id, user2_id,
                       EXTRACT(EPOCH FROM created_at) as ts
                FROM public.matches
            """)
            matches = cur.fetchall()

            # Session outcomes (strongest positive signal)
            cur.execute("""
                SELECT s.match_id, s.proposed_by, s.status,
                       m.user1_id, m.user2_id
                FROM public.sessions s
                JOIN public.matches m ON m.id = s.match_id
            """)
            sessions = cur.fetchall()

        return {
            "swipes": swipes,
            "users": users,
            "matches": matches,
            "sessions": sessions,
        }
    finally:
        conn.close()


def fetch_eligible_pairs(max_distance_miles: int = 100):
    """Fetch all eligible user pairs for batch prediction."""
    conn = get_connection()
    try:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            cur.execute("""
                SELECT a.id as viewer_id, b.id as candidate_id
                FROM public.users a
                CROSS JOIN public.users b
                WHERE a.id != b.id
                  AND a.first_name IS NOT NULL AND a.first_name != ''
                  AND b.first_name IS NOT NULL AND b.first_name != ''
                  AND a.is_test_account = false
                  AND b.is_test_account = false
                  AND a.location IS NOT NULL
                  AND b.location IS NOT NULL
                  AND ST_DWithin(a.location, b.location, %s * 1609.344)
            """, (max_distance_miles,))
            return cur.fetchall()
    finally:
        conn.close()


def fetch_behavioral_signals() -> dict[str, dict]:
    """Fetch per-user behavioral signals from BigQuery analytics events.

    Returns a dict of user_id → behavioral features:
    - avg_swipe_time: average seconds between card_impression and swipe
    - right_swipe_rate: fraction of right swipes
    - sessions_per_day: average sessions per active day
    - avg_message_length: average chat message length
    - chat_response_rate: fraction of matches where user sent a message
    """
    creds_b64 = settings.google_bigquery_credentials_b64.get_secret_value()
    if not creds_b64:
        return {}

    try:
        from google.cloud import bigquery
        from google.oauth2 import service_account

        creds_json = json.loads(base64.b64decode(creds_b64))

        # Write to temp file (google SDK needs a file path)
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            json.dump(creds_json, f)
            creds_path = f.name

        try:
            credentials = service_account.Credentials.from_service_account_file(creds_path)
            client = bigquery.Client(
                project=settings.bigquery_project,
                credentials=credentials,
            )

            query = f"""
            WITH swipe_timing AS (
                -- Average time between seeing a card and swiping
                SELECT
                    imp.user_id,
                    AVG(TIMESTAMP_DIFF(sw.created_at, imp.created_at, SECOND)) as avg_swipe_seconds
                FROM `{settings.bigquery_project}.{settings.bigquery_dataset}.events` imp
                JOIN `{settings.bigquery_project}.{settings.bigquery_dataset}.events` sw
                    ON imp.user_id = sw.user_id
                    AND imp.event_name = 'card_impression'
                    AND sw.event_name = 'discover_swipe'
                    AND JSON_EXTRACT_SCALAR(imp.properties, '$.candidate_id') = JSON_EXTRACT_SCALAR(sw.properties, '$.profile_id')
                WHERE imp.created_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
                GROUP BY imp.user_id
            ),
            swipe_direction AS (
                SELECT
                    user_id,
                    COUNTIF(JSON_EXTRACT_SCALAR(properties, '$.direction') = 'right') / GREATEST(COUNT(*), 1) as right_swipe_rate,
                    COUNT(*) as total_swipes
                FROM `{settings.bigquery_project}.{settings.bigquery_dataset}.events`
                WHERE event_name = 'discover_swipe'
                    AND created_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
                GROUP BY user_id
            ),
            session_activity AS (
                SELECT
                    user_id,
                    COUNT(DISTINCT session_id) / GREATEST(COUNT(DISTINCT DATE(created_at)), 1) as sessions_per_day
                FROM `{settings.bigquery_project}.{settings.bigquery_dataset}.events`
                WHERE created_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
                GROUP BY user_id
            ),
            chat_activity AS (
                SELECT
                    user_id,
                    AVG(CAST(JSON_EXTRACT_SCALAR(properties, '$.message_length') AS FLOAT64)) as avg_message_length
                FROM `{settings.bigquery_project}.{settings.bigquery_dataset}.events`
                WHERE event_name = 'chat_message_sent'
                    AND created_at > TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
                GROUP BY user_id
            )
            SELECT
                COALESCE(sd.user_id, sa.user_id, ca.user_id) as user_id,
                COALESCE(st.avg_swipe_seconds, 0) as avg_swipe_time,
                COALESCE(sd.right_swipe_rate, 0.5) as right_swipe_rate,
                COALESCE(sd.total_swipes, 0) as total_swipes,
                COALESCE(sa.sessions_per_day, 0) as sessions_per_day,
                COALESCE(ca.avg_message_length, 0) as avg_message_length
            FROM swipe_direction sd
            FULL OUTER JOIN swipe_timing st ON sd.user_id = st.user_id
            FULL OUTER JOIN session_activity sa ON sd.user_id = sa.user_id
            FULL OUTER JOIN chat_activity ca ON sd.user_id = ca.user_id
            """

            results = client.query(query).result()

            signals = {}
            for row in results:
                signals[row.user_id] = {
                    "avg_swipe_time": float(row.avg_swipe_time or 0),
                    "right_swipe_rate": float(row.right_swipe_rate or 0.5),
                    "total_swipes": int(row.total_swipes or 0),
                    "sessions_per_day": float(row.sessions_per_day or 0),
                    "avg_message_length": float(row.avg_message_length or 0),
                }
            return signals
        finally:
            os.unlink(creds_path)
    except Exception as e:
        print(f"[BigQuery] Failed to fetch behavioral signals: {e}")
        return {}


def write_pair_scores(scores: list[dict]):
    """Write predicted scores to user_pair_scores table."""
    if not scores:
        return

    conn = get_connection()
    try:
        with conn.cursor() as cur:
            # Upsert scores
            psycopg2.extras.execute_values(
                cur,
                """
                INSERT INTO public.user_pair_scores (viewer_id, candidate_id, ml_score, model_version)
                VALUES %s
                ON CONFLICT (viewer_id, candidate_id)
                DO UPDATE SET ml_score = EXCLUDED.ml_score,
                             model_version = EXCLUDED.model_version,
                             updated_at = now()
                """,
                [(s["viewer_id"], s["candidate_id"], s["score"], s["model_version"]) for s in scores],
            )
            conn.commit()
    finally:
        conn.close()
