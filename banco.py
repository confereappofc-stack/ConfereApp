# banco.py
# -*- coding: utf-8 -*-
from datetime import datetime
from flask_sqlalchemy import SQLAlchemy

db = SQLAlchemy()


# =====================================================
# USUÁRIO
# =====================================================
class Usuario(db.Model):
    __tablename__ = "usuarios"

    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(120), nullable=False)
    email = db.Column(db.String(200), unique=True, index=True, nullable=False)
    senha = db.Column(db.String(255), nullable=True)

    # Perfis / papéis
    role = db.Column(db.String(50), default="user")          # ex: 'user', 'admin', 'master'
    is_admin = db.Column(db.Boolean, default=False)          # flag legado
    perfil = db.Column(db.String(50))                        # ex: 'Master' para admin master

    ativo = db.Column(db.Boolean, default=True)

    # Tipo de usuário: cliente / empresa / técnico
    tipo = db.Column(db.String(30), default="cliente")       # cliente / empresa / técnico
    cpf = db.Column(db.String(20))
    mercado_id = db.Column(db.Integer, db.ForeignKey("mercados.id"))
    criado_em = db.Column(db.DateTime, default=datetime.utcnow)

    # Relações úteis
    lista_itens = db.relationship(
        "ListaItem",
        backref="usuario",
        lazy="dynamic",
        cascade="all, delete-orphan"
    )

    # ---- ADMIN / PERFIS ----
    @property
    def is_master(self) -> bool:
        """
        Admin Master verdadeiro:
        - PERFIL = 'Master' (case-insensitive), ou
        - ROLE = 'master'
        """
        perfil = (self.perfil or '').strip().lower()
        role = (self.role or '').strip().lower()
        return perfil == 'master' or role == 'master'

    @property
    def is_admin_tecnico(self) -> bool:
        """
        Admin técnico: TIPO = 'técnico' (com ou sem acento)
        """
        tipo = (self.tipo or '').strip().lower()
        return tipo in {'técnico', 'tecnico'}

    @property
    def is_admin_any(self) -> bool:
        """
        Considera admin se:
        - for Master
        - ou for técnico
        - ou tiver flag is_admin=True
        - ou role = 'admin'
        """
        role = (self.role or '').strip().lower()
        return (
            self.is_master
            or self.is_admin_tecnico
            or bool(self.is_admin)
            or role == 'admin'
        )

    # ---- CLIENTE / EMPRESA ----
    @property
    def is_cliente(self) -> bool:
        tipo = (self.tipo or '').strip().lower()
        return tipo == 'cliente'

    @property
    def is_empresa(self) -> bool:
        """
        Empresa/mercado: TIPO = 'empresa'
        """
        tipo = (self.tipo or '').strip().lower()
        return tipo == 'empresa'


# =====================================================
# MERCADO
# =====================================================
class Mercado(db.Model):
    __tablename__ = "mercados"

    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(200), nullable=False)
    cnpj = db.Column(db.String(20), index=True)
    email = db.Column(db.String(200))

    logradouro = db.Column(db.String(200))
    numero     = db.Column(db.String(50))
    bairro     = db.Column(db.String(120))
    cidade     = db.Column(db.String(120))
    uf         = db.Column(db.String(2))
    cep        = db.Column(db.String(20))

    # coordenadas
    lat      = db.Column(db.Float)
    lng      = db.Column(db.Float)
    dist_km  = db.Column(db.Float)

    # dono do mercado (empresa dona do cadastro, se houver)
    owner_user_id = db.Column(db.Integer)

    criado_em = db.Column(db.DateTime, default=datetime.utcnow)

    # relação para poder deletar mercado e já apagar preços ligados a ele
    precos = db.relationship(
        "Preco",
        backref="mercado",
        cascade="all, delete-orphan"
    )

    @property
    def endereco_completo(self):
        partes = [
            self.logradouro or "",
            self.numero or "",
            self.bairro or "",
            self.cidade or "",
            self.uf or "",
            self.cep or "",
        ]
        return ", ".join([p for p in partes if p])

    @property
    def google_maps_url(self):
        from urllib.parse import quote_plus
        if self.lat and self.lng:
            return f"https://www.google.com/maps/dir/?api=1&destination={self.lat},{self.lng}"
        q = quote_plus(self.endereco_completo)
        return f"https://www.google.com/maps/search/?api=1&query={q}"

    @property
    def waze_url(self):
        from urllib.parse import quote_plus
        if self.lat and self.lng:
            return f"https://waze.com/ul?ll={self.lat},{self.lng}&navigate=yes"
        q = quote_plus(self.endereco_completo)
        return f"https://waze.com/ul?q={q}&navigate=yes"


# =====================================================
# PRODUTO
# =====================================================
class Produto(db.Model):
    __tablename__ = "produtos"

    id = db.Column(db.Integer, primary_key=True)
    nome = db.Column(db.String(200), nullable=False)
    nome_canonico = db.Column(
        db.String(220), nullable=False, index=True, unique=True
    )
    categoria = db.Column(db.String(80))

    def set_nome(self, nome):
        from utils.text import normalize_str

        self.nome = nome.strip()
        self.nome_canonico = normalize_str(nome)


# =====================================================
# HISTÓRICO DE COMPARAÇÃO
# =====================================================
class HistoricoComparacao(db.Model):
    __tablename__ = "historico_comparacao"

    id = db.Column(db.Integer, primary_key=True)
    usuario_id = db.Column(
        db.Integer,
        db.ForeignKey("usuarios.id"),
        nullable=False,
    )

    lat = db.Column(db.Float, nullable=True)
    lng = db.Column(db.Float, nullable=True)
    raio_km = db.Column(db.Float, nullable=True)

    best_single_total = db.Column(db.Float, nullable=True)
    best_split_total = db.Column(db.Float, nullable=True)
    economia = db.Column(db.Float, nullable=True)

    criado_em = db.Column(db.DateTime, default=datetime.utcnow)

    # relação para acessar o usuário a partir do histórico
    usuario = db.relationship("Usuario", backref="historicos")


# =====================================================
# MINHA LISTA (estrutura antiga – se ainda usada)
# =====================================================
class MinhaLista(db.Model):
    __tablename__ = "minha_lista"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("usuarios.id"),
        index=True,
        nullable=False,
    )
    criada_em = db.Column(db.DateTime, server_default=db.func.now())


# =====================================================
# ITENS DA LISTA (estrutura antiga – se ainda usada)
# =====================================================
class MinhaListaItem(db.Model):
    __tablename__ = "minha_lista_itens"

    id = db.Column(db.Integer, primary_key=True)
    lista_id = db.Column(
        db.Integer,
        db.ForeignKey("minha_lista.id"),
        index=True,
        nullable=False,
    )
    produto_id = db.Column(
        db.Integer,
        db.ForeignKey("produtos.id"),
        index=True,
        nullable=False,
    )
    qtd = db.Column(db.Integer, nullable=False, default=1)

    __table_args__ = (
        db.UniqueConstraint("lista_id", "produto_id", name="uq_lista_produto"),
    )


# =====================================================
# PREÇOS
# =====================================================
class Preco(db.Model):
    __tablename__ = "precos"

    id = db.Column(db.Integer, primary_key=True)
    produto_id = db.Column(
        db.Integer,
        db.ForeignKey("produtos.id"),
        nullable=False,
        index=True,
    )
    mercado_id = db.Column(
        db.Integer,
        db.ForeignKey("mercados.id"),
        nullable=False,
        index=True,
    )
    preco = db.Column(db.Float, nullable=False)
    marca = db.Column(db.String(120))
    unidade = db.Column(db.String(50))
    tamanho = db.Column(db.Float)
    criado_em = db.Column(db.DateTime, default=datetime.utcnow)


# =====================================================
# ITENS DA LISTA DO USUÁRIO (modelo atual usado em /lista)
# =====================================================
class ListaItem(db.Model):
    __tablename__ = "lista_itens"

    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(
        db.Integer,
        db.ForeignKey("usuarios.id"),
        index=True,
        nullable=False,
    )
    produto_id = db.Column(
        db.Integer,
        db.ForeignKey("produtos.id"),
        index=True,
        nullable=False,
    )
    quantidade = db.Column(db.Float, default=1.0)
    marca = db.Column(db.String(120))
    unidade = db.Column(db.String(50))
    tamanho = db.Column(db.Float)
    criado_em = db.Column(db.DateTime, default=datetime.utcnow)


# =====================================================
# GEO CACHE (para não repetir chamadas de geocodificação)
# =====================================================
class GeoCache(db.Model):
    __tablename__ = "geo_cache"

    id = db.Column(db.Integer, primary_key=True)
    chave = db.Column(db.String(255), unique=True, index=True, nullable=False)
    lat = db.Column(db.Float, nullable=False)
    lng = db.Column(db.Float, nullable=False)
    criado_em = db.Column(db.DateTime, default=datetime.utcnow)
