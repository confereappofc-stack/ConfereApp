import sqlite3

DB = 'confereapp.db'
conn = sqlite3.connect(DB)
cur = conn.cursor()

# 1) Garantir tabela mercados
cur.execute("""
CREATE TABLE IF NOT EXISTS mercados (
    id       INTEGER PRIMARY KEY AUTOINCREMENT,
    nome     TEXT,
    dist_km  REAL
)
""")

# 2) Adicionar colunas novas, se faltarem
def add_col(nome, tipo):
    cur.execute("PRAGMA table_info(mercados)")
    cols = [r[1] for r in cur.fetchall()]
    if nome not in cols:
        cur.execute(f"ALTER TABLE mercados ADD COLUMN {nome} {tipo}")

add_col('cnpj',     'TEXT')   # será obrigatório na aplicação
add_col('email',    'TEXT')
add_col('logradouro','TEXT')
add_col('numero',   'TEXT')
add_col('bairro',   'TEXT')
add_col('cidade',   'TEXT')
add_col('uf',       'TEXT')
add_col('cep',      'TEXT')

# 3) Preenche CNPJ "pendente" para registros antigos (para não ficar nulo)
cur.execute("UPDATE mercados SET cnpj = 'PENDENTE_' || id WHERE cnpj IS NULL OR TRIM(cnpj) = ''")

# 4) Índices (UNIQUE parcial em CNPJ real; ignora os 'PENDENTE_%')
cur.execute("CREATE INDEX IF NOT EXISTS idx_mercados_cnpj ON mercados(cnpj)")
# se a sua versão do SQLite suportar índices parciais:
# cur.execute(\"CREATE UNIQUE INDEX IF NOT EXISTS ux_mercados_cnpj_validos ON mercados(cnpj) WHERE cnpj NOT LIKE 'PENDENTE_%'\") 

conn.commit()
conn.close()
print("Migração concluída.")
