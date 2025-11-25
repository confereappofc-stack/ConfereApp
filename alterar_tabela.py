# -*- coding: utf-8 -*-
import sqlite3

conn = sqlite3.connect('confereapp.db')
c = conn.cursor()

# cria a tabela se não existir (já com coluna senha)
c.execute('''
CREATE TABLE IF NOT EXISTS usuarios (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT NOT NULL,
    email TEXT NOT NULL,
    senha TEXT
)
''')

# adiciona coluna senha se o banco antigo não tiver
cols = [r[1] for r in c.execute("PRAGMA table_info(usuarios)").fetchall()]
if 'senha' not in cols:
    c.execute("ALTER TABLE usuarios ADD COLUMN senha TEXT")

conn.commit()
conn.close()
print("OK: tabela 'usuarios' com coluna 'senha'.")

