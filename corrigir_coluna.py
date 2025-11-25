import sqlite3

conn = sqlite3.connect('confereapp.db')
cur = conn.cursor()

# Verifica colunas da tabela 'precos'
cur.execute("PRAGMA table_info(precos)")
cols = [r[1] for r in cur.fetchall()]

# Cria a coluna se não existir (sem DEFAULT)
if 'criado_em' not in cols:
    cur.execute("ALTER TABLE precos ADD COLUMN criado_em TEXT")
    conn.commit()

# Preenche linhas antigas
cur.execute("UPDATE precos SET criado_em = datetime('now') WHERE criado_em IS NULL")
conn.commit()

conn.close()
print("OK: coluna 'criado_em' garantida e preenchida.")
