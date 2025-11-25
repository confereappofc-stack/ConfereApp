# banco.py
# -*- coding: utf-8 -*-
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()

class Usuario(db.Model):
    __tablename__ = "usuarios"
    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(200), unique=True, index=True, nullable=False)
    senha = db.Column(db.String(255), nullable=True)

    # campos “defensivos” que o app.py garante/usa
    role = db.Column(db.String(50), default="user")           # "user" | "admin_tecnico" | "admin_geral"
    is_admin = db.Column(db.Boolean, default=False)
    ativo = db.Column(db.Boolean, default=True)

    # perfis
    tipo = db.Column(db.String(30), default="cliente")        # "cliente" | "empresa"
    cpf = db.Column(db.String(20))
    mercado_id = db.Column(db.Integer, db.ForeignKey("mercados.id"))

    criado_em = db.Column(db.DateTime, default=datetime.utcnow)

    # Helpers para o Jinja
    @property
    def is_admin_any(self):
        return bool(self.role in ("admin_tecnico", "admin_geral") or self.is_admin)

    @property
    def is_cliente(self):
        return (self.tipo or "").lower() == "cliente"

    @property
    def is_empresa(self):
        return (self.tipo or "").lower() == "empresa"


class Mercado(db.Model):
    __tablename__ = "mercados"
    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(200), nullable=False)
    cnpj = db.Column(db.String(20), index=True)
    email = db.Column(db.String(200))

    # endereço
    logradouro = db.Column(db.String(200))
    numero = db.Column(db.String(50))
    bairro = db.Column(db.String(120))
    cidade = db.Column(db.String(120))
    uf = db.Column(db.String(2))
    cep = db.Column(db.String(20))

    # localização opcional
    lat = db.Column(db.Float)
    lng = db.Column(db.Float)
    dist_km = db.Column(db.Float)

    # dono (empresa usuária)
    owner_user_id = db.Column(db.Integer)

    criado_em = db.Column(db.DateTime, default=datetime.utcnow)


class Produto(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(200), nullable=False)
    nome_canonico = db.Column(db.String(220), nullable=False, index=True, unique=True)
    categoria = db.Column(db.String(80))

    def set_nome(self, nome):
        from utils.text import normalize_str
        self.nome = nome.strip()
        self.nome_canonico = normalize_str(nome)

class MinhaLista(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("usuario.id"), index=True, nullable=False)
    criada_em = db.Column(db.DateTime, server_default=db.func.now())

class MinhaListaItem(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    lista_id = db.Column(db.Integer, db.ForeignKey("minha_lista.id"), index=True, nullable=False)
    produto_id = db.Column(db.Integer, db.ForeignKey("produto.id"), index=True, nullable=False)
    qtd = db.Column(db.Integer, nullable=False, default=1)

    __table_args__ = (db.UniqueConstraint("lista_id", "produto_id", name="uq_lista_produto"),)


class Preco(db.Model):
    __tablename__ = "precos"
    id = db.Column(db.Integer, primary_key=True)
    produto_id = db.Column(db.Integer, db.ForeignKey("produtos.id"), nullable=False, index=True)
    mercado_id = db.Column(db.Integer, db.ForeignKey("mercados.id"), nullable=False, index=True)

    preco = db.Column(db.Float, nullable=False)

    # variantes
    marca = db.Column(db.String(120))
    unidade = db.Column(db.String(50))
    tamanho = db.Column(db.Float)

    criado_em = db.Column(db.DateTime, default=datetime.utcnow)


class ListaItem(db.Model):
    __tablename__ = "lista_itens"
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey("usuarios.id"), index=True, nullable=False)
    produto_id = db.Column(db.Integer, db.ForeignKey("produtos.id"), index=True, nullable=False)
    quantidade = db.Column(db.Float, default=1.0)

    # variantes salvas na lista
    marca = db.Column(db.String(120))
    unidade = db.Column(db.String(50))
    tamanho = db.Column(db.Float)

    criado_em = db.Column(db.DateTime, default=datetime.utcnow)
