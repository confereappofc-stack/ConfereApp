# -*- coding: utf-8 -*-
import sqlite3

conn = sqlite3.connect('confereapp.db')
c = conn.cursor()

# Mercados (supermercados)
c.execute('''
CREATE TABLE IF NOT EXISTS mercados(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  nome TEXT NOT NULL
);
''')

# Produtos (catálogo)
c.execute('''
CREATE TABLE IF NOT EXISTS produtos(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  nome TEXT NOT NULL
);
''')

# Preços por mercado (produto x mercado x preço)
c.execute('''
CREATE TABLE IF NOT EXISTS precos(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  produto_id INTEGER NOT NULL,
  mercado_id INTEGER NOT NULL,
  preco REAL NOT NULL,
  FOREIGN KEY(produto_id) REFERENCES produtos(id),
  FOREIGN KEY(mercado_id) REFERENCES mercados(id)
);
''')

# Itens de uma lista do usuário (lista simples)
c.execute('''
CREATE TABLE IF NOT EXISTS listas(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  usuario_email TEXT NOT NULL,
  criado_em TEXT DEFAULT (datetime('now','localtime'))
);
''')

c.execute('''
CREATE TABLE IF NOT EXISTS lista_itens(
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  lista_id INTEGER NOT NULL,
  produto_id INTEGER NOT NULL,
  quantidade REAL DEFAULT 1,
  FOREIGN KEY(lista_id) REFERENCES listas(id),
  FOREIGN KEY(produto_id) REFERENCES produtos(id)
);
''')

conn.commit()
conn.close()
print("OK: estruturas criadas/atualizadas.")
