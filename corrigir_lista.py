# corrigir_lista.py
import sqlite3

db = "confereapp.db"
conn = sqlite3.connect(db)
cur = conn.cursor()

# Verifica as colunas atuais
cur.execute("PRAGMA table_info(lista_itens)")
info = cur.fetchall()
cols = [r[1] for r in info]

if 'lista_id' in cols:
    print(">> Encontrada coluna indevida 'lista_id'. Ajustando tabela...")

    # Cria tabela nova com o schema correto
    cur.execute("""
        CREATE TABLE IF NOT EXISTS lista_itens_new (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER NOT NULL,
            produto_id INTEGER NOT NULL,
            quantidade REAL NOT NULL DEFAULT 1,
            criado_em TEXT DEFAULT (datetime('now'))
        )
    """)

    # Migra dados existentes (best-effort)
    tem = set(cols)
    if {'id','user_id','produto_id','quantidade','criado_em'}.issubset(tem):
        cur.execute("""
            INSERT INTO lista_itens_new (id, user_id, produto_id, quantidade, criado_em)
            SELECT id, user_id, produto_id, quantidade, criado_em
            FROM lista_itens
        """)
    else:
        cur.execute("""
            INSERT INTO lista_itens_new (user_id, produto_id, quantidade)
            SELECT user_id, produto_id, quantidade
            FROM lista_itens
        """)

    # Troca as tabelas
    cur.execute("DROP TABLE lista_itens")
    cur.execute("ALTER TABLE lista_itens_new RENAME TO lista_itens")
    conn.commit()
    print(">> Tabela 'lista_itens' corrigida com sucesso.")
else:
    print(">> 'lista_itens' já está no formato correto. Nada a fazer.")

conn.close()
