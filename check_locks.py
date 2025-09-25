#!/usr/bin/env python3
"""Check for locks and transactions in TimescaleDB"""

import psycopg2

def check_locks():
    conn = psycopg2.connect(
        host="localhost",
        port=5433,
        user="postgres",
        password="postgres",
        dbname="vectordb"
    )

    with conn.cursor() as cur:
        # Check active connections and transactions
        print("🔍 Active connections:")
        cur.execute("""
            SELECT pid, usename, application_name, state, state_change, query_start, query
            FROM pg_stat_activity
            WHERE datname = 'vectordb' AND state != 'idle'
            ORDER BY query_start;
        """)
        for row in cur.fetchall():
            print(f"  PID {row[0]}: {row[1]} | {row[2]} | {row[3]} | Query: {row[6][:100]}...")

        print("\n🔒 Table locks:")
        cur.execute("""
            SELECT l.locktype, l.database, l.relation::regclass, l.page, l.tuple, l.virtualxid,
                   l.transactionid, l.classid, l.objid, l.objsubid, l.virtualtransaction, l.pid,
                   l.mode, l.granted, a.usename, a.query, a.state
            FROM pg_locks l
            LEFT JOIN pg_stat_activity a ON l.pid = a.pid
            WHERE l.relation = 'vector_embeddings'::regclass OR l.locktype = 'relation'
            ORDER BY l.pid;
        """)
        locks = cur.fetchall()
        if locks:
            for lock in locks:
                print(f"  {lock}")
        else:
            print("  No locks found")

        print("\n⏱️  Long running queries:")
        cur.execute("""
            SELECT pid, now() - pg_stat_activity.query_start AS duration, query, state
            FROM pg_stat_activity
            WHERE (now() - pg_stat_activity.query_start) > interval '1 minutes'
                AND state != 'idle';
        """)
        long_queries = cur.fetchall()
        if long_queries:
            for query in long_queries:
                print(f"  PID {query[0]}: {query[1]} | {query[3]} | {query[2][:100]}...")
        else:
            print("  No long running queries")

    conn.close()

if __name__ == "__main__":
    check_locks()