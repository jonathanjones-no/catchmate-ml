"""Database connection and data fetching for training/prediction."""

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
