# fix_columns.py
import sqlite3
DB = "confereapp.db"  # ajuste se seu arquivo tiver outro nome

con = sqlite3.connect(DB)
cur = con.cursor()

# verifica colunas existentes
cur.execute("PRAGMA table_info(usuarios)")
cols = {row[1] for row in cur.fetchall()}

# adiciona colunas que faltam
if "is_admin" not in cols:
    print("Adicionando coluna is_admin...")
    cur.execute("ALTER TABLE usuarios ADD COLUMN is_admin INTEGER DEFAULT 0")

if "ativo" not in cols:
    print("Adicionando coluna ativo...")
    cur.execute("ALTER TABLE usuarios ADD COLUMN ativo INTEGER DEFAULT 1")

con.commit()
con.close()
print("OK. Estrutura atualizada.")

