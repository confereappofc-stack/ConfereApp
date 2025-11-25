# -*- coding: utf-8 -*-
import sqlite3

conn = sqlite3.connect('confereapp.db')
c = conn.cursor()

# adiciona colunas se não existirem
cols = [r[1] for r in c.execute("PRAGMA table_info(precos)").fetchall()]

if 'marca' not in cols:
    c.execute("ALTER TABLE precos ADD COLUMN marca TEXT")

if 'tamanho' not in cols:
    c.execute("ALTER TABLE precos ADD COLUMN tamanho REAL")

if 'unidade' not in cols:
    c.execute("ALTER TABLE precos ADD COLUMN unidade TEXT")

conn.commit()
conn.close()
print("OK: tabela 'precos' agora tem marca, tamanho e unidade.")
