import sqlite3

import config as _

# Cria/conecta ao banco e ativa o modo WAL
conn = sqlite3.connect(_.SQLITE_MEMORY_DATABASE)
conn.execute("PRAGMA journal_mode=WAL;")
conn.close()

print(f"Banco {_.SQLITE_MEMORY_DATABASE} configurado para modo WAL (alta concorrência).")
