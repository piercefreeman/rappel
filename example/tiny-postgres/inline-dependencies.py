# /// script
# dependencies = ["psycopg2-binary"]
# ///

import psycopg2

conn = psycopg2.connect("postgresql://postgres:pass@localhost/postgres")
cur = conn.cursor()

cur.execute("CREATE TABLE demo (word TEXT);")
cur.execute("INSERT INTO demo VALUES ('hello'), ('world');")
conn.commit()

cur.execute("SELECT * FROM demo;")
for row in cur.fetchall():
    print(row[0])

conn.close()
