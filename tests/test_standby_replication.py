"""Integration test for standby replication via pgBackRest WAL archiving.

Verify that a standby PostgreSQL instance receives
and replays WAL from the primary, so that data written to the primary
appears on the standby after a WAL switch.

The standby connects to the same MinIO object stores as the primary and
uses pgBackRest's restore_command to fetch archived WAL segments.

Requirements:
    - Docker Compose environment running with postgres, postgres-standby,
      minio-site1, minio-site2
    - Primary archiving enabled (FASTDB_ARCHIVE_ENABLED=true)
    - Standby started and in recovery mode
    - elasticc2_test_data/ directory present (used by snana_fits_maintables_loaded_module)

Run with:
    pytest tests/test_standby_replication.py -v -s

"""

import os
import time

import psycopg
import pytest

from db import DB

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

STANDBY_HOST = os.environ.get("FASTDB_TEST_STANDBY_HOST", "postgres-standby")
STANDBY_PORT = int(os.environ.get("FASTDB_TEST_STANDBY_PORT", "5432"))
STANDBY_USER = "postgres"
STANDBY_DBNAME = "fastdb"

# How long to wait for WAL replay on the standby (seconds).
REPLICATION_TIMEOUT = int(os.environ.get("FASTDB_TEST_REPLICATION_TIMEOUT", "120"))
REPLICATION_POLL_INTERVAL = 3

# Path to the secrets directory (same layout as the primary)
SECRETS_DIR = os.environ.get(
    "FASTDB_TEST_SECRETS_DIR",
    os.path.join(os.path.dirname(__file__), "secrets"),
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _get_standby_connection():
    """Return a psycopg connection to the standby database.

    Reads the postgres password from the secrets directory (same file
    the primary uses).
    """
    password_file = os.path.join(SECRETS_DIR, "pgpasswd")
    if os.path.exists(password_file):
        with open(password_file) as f:
            password = f.read().strip()
    else:
        password = "postgres"

    return psycopg.connect(
        host=STANDBY_HOST,
        port=STANDBY_PORT,
        user=STANDBY_USER,
        password=password,
        dbname=STANDBY_DBNAME,
    )


def _standby_is_in_recovery() -> bool:
    """Check that the standby is running in recovery (standby) mode."""
    conn = _get_standby_connection()
    try:
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute("SELECT pg_is_in_recovery();")
        return cursor.fetchone()[0] is True
    finally:
        conn.close()


def _query_standby_count(table: str) -> int:
    """Return the row count of a table on the standby."""
    conn = _get_standby_connection()
    try:
        conn.autocommit = True
        cursor = conn.cursor()
        cursor.execute(f"SELECT COUNT(*) FROM {table}")
        return cursor.fetchone()[0]
    finally:
        conn.close()


def _force_wal_switch():
    """Force the primary to switch WAL segments so archiving fires."""
    with DB() as conn:
        cursor = conn.cursor()
        cursor.execute("SELECT pg_switch_wal();")
        conn.commit()


def _wait_for_standby_count(table: str, expected: int) -> int:
    """Poll the standby until the row count for *table* reaches *expected*, or timeout.

    Returns the final observed count.
    """
    deadline = time.time() + REPLICATION_TIMEOUT
    count = 0
    while time.time() < deadline:
        try:
            count = _query_standby_count(table)
            if count >= expected:
                return count
        except Exception:
            # Standby may momentarily reject connections during WAL replay
            pass
        time.sleep(REPLICATION_POLL_INTERVAL)
    return count


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestStandbyReplication:
    """Verify that a standby receives and replays WAL from the primary.

    Uses the snana_fits_maintables_loaded_module fixture to load real
    ELAsTiCC2 data on the primary, then confirms the same rows appear
    on the standby after WAL archiving and replay.
    """

    # -- Pre-conditions ----------------------------------------------------

    def test_standby_is_running(self):
        """The standby must be reachable and accepting connections."""
        conn = _get_standby_connection()
        conn.close()

    def test_standby_is_in_recovery(self):
        """The standby must be in recovery (standby) mode, not primary."""
        assert _standby_is_in_recovery(), (
            "Standby is NOT in recovery mode. "
            "It may have been promoted or started as a primary."
        )

    def test_primary_is_not_in_recovery(self):
        """The primary must NOT be in recovery mode."""
        with DB() as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT pg_is_in_recovery();")
            in_recovery = cursor.fetchone()[0]
        assert not in_recovery, "Primary is in recovery mode — something is wrong."

    # -- Core replication test ---------------------------------------------

    def test_snana_data_replicated_to_standby(
        self, snana_fits_maintables_loaded_module
    ):
        """Load ELAsTiCC2 data on primary, verify it appears on the standby.

        Steps:
            1. The fixture loads ~54k rows on the primary.
            2. Force a WAL switch so the segments get archived.
            3. Poll the standby until row counts match (or timeout).
        """
        nobj, nsrc, nfrc, _nhost = snana_fits_maintables_loaded_module

        # Verify the primary has the expected data
        assert nobj == 346
        assert nsrc == 1862
        assert nfrc == 52172
        # Remove host galaxy check

        # Force WAL switch to trigger archiving
        _force_wal_switch()

        # Wait for replication and check each table
        tables = {
            "diaobject": nobj,
            "diasource": nsrc,
            "diaforcedsource": nfrc,
        }

        results = {}
        for table, expected in tables.items():
            count = _wait_for_standby_count(table, expected)
            results[table] = count
            print(f"\n  {table}: primary={expected}, standby={count}")

        # Assert all tables replicated
        for table, expected in tables.items():
            assert results[table] == expected, (
                f"Table '{table}' has {results[table]} rows on standby, "
                f"expected {expected}. Replication may be lagging or broken "
                f"(waited {REPLICATION_TIMEOUT}s)."
            )

    # -- Additional replication checks -------------------------------------

    def test_incremental_write_replicates(self):
        """An additional write on the primary must also appear on the standby.

        This catches issues where the initial restore works but ongoing
        WAL replay is broken (e.g., restore_command misconfiguration,
        credential issues on repeated fetches).
        """
        test_table = "replication_check"

        # Write on primary
        with DB() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"CREATE TABLE IF NOT EXISTS {test_table} "
                f"(id serial PRIMARY KEY, msg text, ts timestamp DEFAULT now())"
            )
            cursor.execute(
                f"INSERT INTO {test_table} (msg) VALUES ('incremental replication test')"
            )
            conn.commit()

        _force_wal_switch()

        # Poll standby for the new table and row
        deadline = time.time() + REPLICATION_TIMEOUT
        found = False
        while time.time() < deadline:
            try:
                conn = _get_standby_connection()
                conn.autocommit = True
                cursor = conn.cursor()
                cursor.execute(f"SELECT msg FROM {test_table}")
                rows = cursor.fetchall()
                conn.close()
                if rows and rows[0][0] == "incremental replication test":
                    found = True
                    break
            except Exception:
                pass
            time.sleep(REPLICATION_POLL_INTERVAL)

        assert found, (
            f"Incremental write to '{test_table}' did not replicate to standby "
            f"within {REPLICATION_TIMEOUT}s."
        )
        print("\n  Incremental write replicated successfully")

    def test_standby_rejects_writes(self):
        """The standby must reject write operations (it's read-only)."""
        conn = _get_standby_connection()
        try:
            conn.autocommit = True
            cursor = conn.cursor()
            with pytest.raises(psycopg.errors.ReadOnlySqlTransaction):
                cursor.execute(
                    "CREATE TABLE standby_write_test (id serial)"
                )
        finally:
            conn.close()
