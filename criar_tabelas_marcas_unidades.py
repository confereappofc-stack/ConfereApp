import sqlite3

# Conectar ao banco
conn = sqlite3.connect('confereapp.db')
cursor = conn.cursor()

# Criar tabela de marcas (se não existir)
cursor.execute("""
CREATE TABLE IF NOT EXISTS marcas (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    nome TEXT UNIQUE NOT NULL
)
""")

# Criar tabela de unidades (se não existir)
cursor.execute("""
CREATE TABLE IF NOT EXISTS unidades (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    descricao TEXT UNIQUE NOT NULL
)
""")

conn.commit()
conn.close()

print("Tabelas 'marcas' e 'unidades' criadas/garantidas com sucesso!")
