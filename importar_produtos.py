# -*- coding: utf-8 -*-
import pandas as pd
import sqlite3
import sys
import os

ARQUIVO = None
if len(sys.argv) >= 2:
    ARQUIVO = sys.argv[1]  # permite passar o caminho por argumento
else:
    # altere aqui se preferir fixo:
    ARQUIVO = "lista_supermercado_bom_preco.xlsx"

if not os.path.exists(ARQUIVO):
    print(f"ERRO: arquivo não encontrado: {ARQUIVO}")
    sys.exit(1)

def to_float(x):
    if pd.isna(x):
        return None
    x = str(x).strip().replace(',', '.')
    return float(x)

conn = sqlite3.connect('confereapp.db')
cur = conn.cursor()

from utils.text import normalize_str  # garanta que exista entre os imports do arquivo


def get_or_create_produto(nome: str) -> int:
    """
    Versão usada pelo script importar_produtos.py,
    mesma lógica da função do app principal.
    """
    nome_limpo = (nome or "").strip()
    if not nome_limpo:
        raise ValueError("Nome de produto vazio em get_or_create_produto().")

    nome_canonico = normalize_str(nome_limpo)

    prod = Produto.query.filter_by(nome_canonico=nome_canonico).first()
    if prod:
        return prod.id

    novo = Produto()
    novo.set_nome(nome_limpo)
    db.session.add(novo)
    db.session.flush()

    return novo.id

def get_or_create_mercado(nome):
    cur.execute("SELECT id FROM mercados WHERE nome = ?", (nome,))
    r = cur.fetchone()
    if r: return r[0]
    cur.execute("INSERT INTO mercados (nome) VALUES (?)", (nome,))
    conn.commit()
    return cur.lastrowid

# Lê Excel
df = pd.read_excel(ARQUIVO)
# Normaliza nomes de colunas
df.columns = [str(c).strip() for c in df.columns]

# Aceita tanto "Preço" quanto "Preço (R$)"
col_preco = "Preço (R$)" if "Preço (R$)" in df.columns else "Preço"

obrig = ["Produto", "Marca", "Tamanho", "Unidade", "Mercado", col_preco]
faltando = [c for c in obrig if c not in df.columns]
if faltando:
    print(f"ERRO: faltam colunas no Excel: {faltando}")
    sys.exit(1)

lidas = 0
inseridas = 0
erros = 0

for idx, row in df.iterrows():
    lidas += 1
    try:
        produto = str(row["Produto"]).strip()
        marca   = str(row["Marca"]).strip()
        unidade = str(row["Unidade"]).strip()
        mercado = str(row["Mercado"]).strip()
        tamanho = to_float(row["Tamanho"])
        preco   = to_float(row[col_preco])

        if not produto or not marca or not unidade or not mercado or tamanho is None or preco is None:
            erros += 1
            continue

        pid = get_or_create_produto(produto)
        mid = get_or_create_mercado(mercado)

        cur.execute("""
            INSERT INTO precos (produto_id, mercado_id, preco, marca, tamanho, unidade)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (pid, mid, preco, marca, tamanho, unidade))
        inseridas += 1

    except Exception as e:
        erros += 1
        print(f"Linha {idx+2}: {e}")

conn.commit()
conn.close()

print("==== Resumo ====")
print(f"Linhas lidas: {lidas}")
print(f"Preços inseridos: {inseridas}")
print(f"Erros: {erros}")
