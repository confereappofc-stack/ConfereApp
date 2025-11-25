import sqlite3

# Conecta ao banco de dados (ou cria se não existir)
conn = sqlite3.connect('confereapp.db')
cursor = conn.cursor()

# Cria a tabela de usuários
cursor.execute('''
    CREATE TABLE IF NOT EXISTS usuarios (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        nome TEXT NOT NULL,
        email TEXT NOT NULL
    )
''')

# Salva as alterações e fecha a conexão
conn.commit()
conn.close()

print("Tabela 'usuarios' criada com sucesso.")
