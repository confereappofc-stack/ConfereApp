# -*- coding: utf-8 -*-
import os, io, csv, re, smtplib
from email.mime.text import MIMEText
from datetime import datetime, timedelta
from functools import wraps
from math import radians, cos, ceil, sin, asin, sqrt
import unicodedata


# -----------------------------------------
# Fallback de coordenadas por cidade
# -----------------------------------------
CITY_FALLBACK_COORDS = {
    "toritama": (-8.01021, -36.05540),
    "santa cruz do capibaribe": (-7.9580, -36.2050),
    "surubim": (-7.83676, -35.7547),
    # Adicione outras cidades conforme necessário
}


from flask import Flask, make_response, render_template, request, session, flash, redirect, url_for, send_file

from werkzeug.utils import secure_filename

from sqlalchemy import and_, or_, text, func, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, ProgrammingError

from utils.geo import geocode_with_cache
from banco import Mercado

from utils.text import normalize_str

from collections import defaultdict, namedtuple
from types import SimpleNamespace

from banco import (
    db, Usuario, Mercado, Produto, Preco,
    ListaItem, MinhaLista, MinhaListaItem
)



# Caminho da logo
LOGO_PATH = os.path.join('static', 'img', 'confereapp_logo.png')

# -------------------------
# IMPORTS DO REPORTLAB (ÚNICOS PERMITIDOS)
# -------------------------


def gerar_pdf(html):
    response = make_response()
    response.headers["Content-Type"] = "application/pdf"
    response.headers["Content-Disposition"] = "attachment; filename=relatorio.pdf"

    pisa.CreatePDF(html, dest=response)
    return "PDF temporariamente indisponível"

def _make_pdf_table(data):
    """Cria uma tabela estilizada (modelo B) para o PDF."""
    table = Table(data, repeatRows=1)
    table.setStyle(TableStyle([
        ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#333333')),
        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
        ('ALIGN', (0, 0), (-1, 0), 'CENTER'),

        ('FONTNAME', (0, 1), (-1, -1), 'Helvetica'),
        ('FONTSIZE', (0, 0), (-1, -1), 8),

        # alinhar números à direita
        ('ALIGN', (4, 1), (-1, -1), 'RIGHT'),

        ('GRID', (0, 0), (-1, -1), 0.25, colors.grey),
        ('ROWBACKGROUNDS', (0, 1), (-1, -1),
         [colors.white, colors.HexColor('#f2f2f2')]),
    ]))
    return table


def _to_float(x):
    """
    Converte de forma segura para float.
    Aceita vírgula ou ponto.
    Retorna None se vazio ou inválido.
    """
    try:
        if x is None:
            return None
        s = str(x).strip()
        if not s:
            return None
        return float(s.replace(',', '.'))
    except:
        return None

def _clean_str(s):
    """
    Devolve string limpa (sem espaços); se vier None, devolve "".
    """
    return (s or "").strip()

def _flt(x, default=None):
    """
    Converte para float de forma segura.
    Se não conseguir, devolve o default.
    """
    try:
        return float(x)
    except (TypeError, ValueError):
        return default

def _item_matches_price(it, preco_row):
    """
    Verifica se um registro de preço atende aos filtros do item da lista:
    - Se o item tiver marca preenchida, só aceita aquela marca.
    - Se tiver unidade preenchida, só aceita aquela unidade.
    - Se tiver tamanho preenchido, compara numericamente.
    """
    # Marca
    if _clean_str(getattr(it, "marca", None)):
        if _clean_str(preco_row.marca) != _clean_str(it.marca):
            return False

    # Unidade
    if _clean_str(getattr(it, "unidade", None)):
        if _clean_str(preco_row.unidade).lower() != _clean_str(it.unidade).lower():
            return False

    # Tamanho
    if getattr(it, "tamanho", None) is not None:
        it_t = _flt(it.tamanho, None)
        pr_t = _flt(preco_row.tamanho, None)
        if it_t is None or pr_t is None:
            return False
        if abs(it_t - pr_t) > 1e-9:
            return False

    return True

def _canon_name(s):
    """
    Normaliza um nome de produto para uma forma canônica:
    - remove acentos
    - deixa minúsculo
    - remove caracteres especiais
    - remove espaços extras
    """
    if not s:
        return ""

    # remover acentos
    s = unicodedata.normalize("NFKD", s)
    s = "".join(c for c in s if not unicodedata.combining(c))

    # manter apenas letras e números
    s = "".join(c.lower() for c in s if c.isalnum() or c.isspace())

    return s.strip()


def _haversine_km(lat1, lng1, lat2, lng2):
    """Distância em KM (haversine). Retorna None se faltar algo."""
    if None in (lat1, lng1, lat2, lng2):
        return None
    R = 6371.0
    dlat = radians(lat2 - lat1)
    dlng = radians(lng2 - lng1)
    a = sin(dlat/2)**2 + cos(radians(lat1))*cos(radians(lat2))*sin(dlng/2)**2
    c = 2 * asin(sqrt(a))
    return R * c

haversine_km = _haversine_km

def _dist_km_or_zero(user_lat, user_lng, mercado):
    """
    Calcula a distancia entre o usuario e o mercado usando haversine_km.
    Retorna 0.0 se faltar informação.
    """
    if user_lat is None or user_lng is None:
        return 0.0
    if mercado.lat is None or mercado.lng is None:
        return 0.0
    try:
        return float(haversine_km(user_lat, user_lng, mercado.lat, mercado.lng) or 0.0)
    except:
        return 0.0


def _build_offers_for_items(produto_ids, mercado_ids):
    """
    Para cada (produto, mercado), pega a oferta mais barata
    (menor preço) entre os registros de Preco.

    Retorna:
      best_variant[(pid, mid)] = {
        "preco": float,
        "marca": str,
        "unidade": str,
        "tamanho": float
      }
    """
    best_variant = {}
    if not produto_ids or not mercado_ids:
        return best_variant

    precos = (
        Preco.query
        .filter(
            Preco.produto_id.in_(produto_ids),
            Preco.mercado_id.in_(mercado_ids)
        )
        .all()
    )

    for p in precos:
        key = (int(p.produto_id), int(p.mercado_id))
        preco = _to_float(p.preco)
        if preco is None:
            continue

        cur = best_variant.get(key)
        if cur is None or preco < cur["preco"]:
            best_variant[key] = {
                "preco": preco,
                "marca": (p.marca or "").strip(),
                "unidade": (p.unidade or "").strip(),
                "tamanho": _to_float(p.tamanho) or 0.0,
            }

    return best_variant


def _solve_allocation_ilp(
    itens_lista,
    mercados,
    best_variant,
    distances,
    max_markets=3,
    lambda_per_market=0.0,
    alpha_per_km=0.0
):
    if pl is None:
        # fallback caso PuLP não exista
        if max_markets == 1:
            best_total = None
            best_mid = None
            for m in mercados:
                mid = int(m.id)
                ok = True
                total = 0.0
                for pid, qtd in itens_lista:
                    v = best_variant.get((pid, mid))
                    if not v:
                        ok = False
                        break
                    total += qtd * v["preco"]
                if ok and (best_total is None or total < best_total):
                    best_total = total
                    best_mid = mid
            allocation = {}
            used = set()
            if best_mid is not None:
                used.add(best_mid)
                for pid, qtd in itens_lista:
                    v = best_variant[(pid, best_mid)]
                    allocation[pid] = {**v, "mercado_id": best_mid, "qtd": qtd}
            return allocation, used, (best_total or 0.0)
        else:
            allocation = {}
            used = set()
            total = 0.0
            for pid, qtd in itens_lista:
                escolha = None
                for m in mercados:
                    mid = int(m.id)
                    v = best_variant.get((pid, mid))
                    if not v:
                        continue
                    if escolha is None or v["preco"] < escolha["preco"]:
                        escolha = v | {"mercado_id": mid}
                if escolha:
                    allocation[pid] = escolha | {"qtd": qtd}
                    used.add(escolha["mercado_id"])
                    total += qtd * escolha["preco"]
            return allocation, used, total

    # MODELO ILP (com PuLP)
    prob = pl.LpProblem("confereapp_min_compra", pl.LpMinimize)

    mercados_ids = [int(m.id) for m in mercados]
    item_ids = [int(pid) for (pid, _q) in itens_lista]

    x = pl.LpVariable.dicts(
        "x",
        [(i, m) for i in item_ids for m in mercados_ids],
        lowBound=0,
        upBound=1,
        cat=pl.LpBinary,
    )

    y = pl.LpVariable.dicts(
        "y",
        mercados_ids,
        lowBound=0,
        upBound=1,
        cat=pl.LpBinary,
    )

    cost_terms = []
    for (pid, qtd) in itens_lista:
        for mid in mercados_ids:
            v = best_variant.get((pid, mid))
            if v:
                cost_terms.append(qtd * v["preco"] * x[(pid, mid)])

    if lambda_per_market != 0.0:
        for mid in mercados_ids:
            cost_terms.append(lambda_per_market * y[mid])

    if alpha_per_km != 0.0:
        for mid in mercados_ids:
            dk = float(distances.get(mid, 0.0))
            cost_terms.append(alpha_per_km * dk * y[mid])

    prob += pl.lpSum(cost_terms)

    for (pid, _qtd) in itens_lista:
        prob += pl.lpSum([x[(pid, mid)] for mid in mercados_ids if best_variant.get((pid, mid))]) == 1

    for (pid, _qtd) in itens_lista:
        for mid in mercados_ids:
            if best_variant.get((pid, mid)):
                prob += x[(pid, mid)] <= y[mid]
            else:
                prob += x[(pid, mid)] == 0

    prob += pl.lpSum([y[mid] for mid in mercados_ids]) <= int(max_markets)

    prob.solve(pl.PULP_CBC_CMD(msg=False))

    allocation = {}
    used = set()
    total = 0.0
    for (pid, qtd) in itens_lista:
        chosen_mid = None
        for mid in mercados_ids:
            if x[(pid, mid)].value() > 0.5:
                chosen_mid = mid
                break
        if chosen_mid:
            v = best_variant[(pid, chosen_mid)]
            allocation[pid] = {
                "mercado_id": chosen_mid,
                "preco": v["preco"],
                "marca": v["marca"],
                "unidade": v["unidade"],
                "tamanho": v["tamanho"],
                "qtd": qtd,
            }
            used.add(chosen_mid)
            total += qtd * v["preco"]

    return allocation, used, total


try:
    import pulp as pl  # ILP
except ImportError:
    pl = None

# === Config do app / banco ===
app = Flask(__name__)
app.secret_key = os.getenv("SECRET_KEY", "confereapp_secret")

BASE_DIR = os.path.dirname(__file__)
DB_PATH = os.path.join(BASE_DIR, "confereapp.db")
app.config["SQLALCHEMY_DATABASE_URI"] = f"sqlite:///{DB_PATH}"
app.config["SQLALCHEMY_TRACK_MODIFICATIONS"] = False

db.init_app(app)

# Cache simples em memória para armazenar o resultado da comparação
# por usuário. Em ambiente de produção isso seria diferente, mas
# para seu uso local funciona bem.
comparar_cache = {}


app.config['UPLOAD_FOLDER'] = os.path.join(os.path.dirname(__file__), 'uploads')
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

ALLOWED_EXTENSIONS = {'xlsx'}



# === login_required simples baseado em session['user_id'] ===
def login_required(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        if 'user_id' not in session:
            flash('Please log in to access this page.', 'warning')
            return redirect(url_for('login', next=request.path))
        return f(*args, **kwargs)
    return decorated_function


def _u_strip(v):
    return str(v).strip() if v is not None else ""

def to_float(v):
    try:
        if v is None: return None
        s = str(v).replace("R$", "").replace(".", "").replace(",", ".").strip()
        return float(s)
    except:
        return None

# >>> Resolve acentos SEM mudar o banco agora

def _norm_product_name(nome):
    # Mantemos seu fluxo atual (nome “bonito”); a comparação acento-insensível é no helper acima
    return (nome or "").strip()

def _norm_product_name(nome):
    """Normaliza nome para comparação (mantém acentos, mas tira variações invisíveis)"""
    nome = _u_strip(nome)
    return nome if nome else None



def looks_like_cnpj(cnpj_raw):
    """Verifica formato simples do CNPJ (somente numérico e tamanho)"""
    if not cnpj_raw:
        return False
    c = "".join([d for d in cnpj_raw if d.isdigit()])
    return len(c) == 14


from flask_sqlalchemy import SQLAlchemy
from flask_migrate import Migrate
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime
from flask import flash
from dotenv import load_dotenv
load_dotenv()

from collections import defaultdict
from itsdangerous import URLSafeTimedSerializer, BadSignature, SignatureExpired
from sqlalchemy import text, event
from sqlalchemy.engine import Engine
from sqlalchemy.exc import OperationalError, ProgrammingError
from flask import g

def _norm_brand(s):
    if not s: return ''
    return s.strip().title()           # “soya” -> “Soya”

def _norm_product(x: str) -> str:
    s = (x or "").strip()
    s = " ".join(s.split())  # colapsa múltiplos espaços em 1
    return s

def _norm_unit(s):
    if not s: return ''
    return s.strip().lower()           # “KG” -> “kg”

def _norm_size(x):
    # None, '', '0', 0, '0.0' viram None; senão vira float
    if x in (None, '', '0', '0.0', 0, 0.0): 
        return None
    try:
        return float(str(x).replace(',', '.'))
    except Exception:
        return None



# -----------------------------------------------------------------------------
# PRAGMA CASCADE no SQLite
# -----------------------------------------------------------------------------
@event.listens_for(Engine, "connect")
def set_sqlite_pragma(dbapi_connection, connection_record):
    try:
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()
    except Exception:
        pass


@app.before_request
def _load_current_user():
    uid = session.get('user_id')
    g.current_user = db.session.get(Usuario, uid) if uid else None

# -----------------------------------------------------------------------------
# SMTP helpers
# -----------------------------------------------------------------------------
def _smtp_config():
    host = os.getenv("SMTP_HOST")
    port = os.getenv("SMTP_PORT")
    user = os.getenv("SMTP_USER")
    pwd  = os.getenv("SMTP_PASS")
    from_addr = os.getenv("SMTP_FROM") or user
    if host and port and user and pwd and from_addr:
        try:
            port = int(port)
        except ValueError:
            port = 587
        return {"host": host, "port": port, "user": user, "pwd": pwd, "from": from_addr}
    return None


def enviar_reset_email(destinatario: str, reset_url: str) -> bool:
    assunto = "Redefinição de senha - ConfereApp"
    corpo_html = f"""
    <p>Olá,</p>
    <p>Recebemos uma solicitação para redefinir sua senha.</p>
    <p>Clique no link abaixo para criar uma nova senha:</p>
    <p><a href="{reset_url}" target="_blank">{reset_url}</a></p>
    <p>Se você não solicitou, ignore este e-mail.</p>
    """
    cfg = _smtp_config()
    if not cfg:
        # modo dev: loga no console
        print("=== [DEV] E-mail de reset (simulado) ===")
        print("Para:", destinatario)
        print("Assunto:", assunto)
        print("Link:", reset_url)
        print("========================================")
        return True

    try:
        msg = MIMEText(corpo_html, "html", "utf-8")
        msg["Subject"] = assunto
        msg["From"] = cfg["from"]
        msg["To"] = destinatario

        with smtplib.SMTP(cfg["host"], cfg["port"]) as server:
            server.ehlo()
            if cfg["port"] in (587, 25):
                server.starttls()
                server.ehlo()
            server.login(cfg["user"], cfg["pwd"])
            server.sendmail(cfg["from"], [destinatario], msg.as_string())
        return True
    except Exception as e:
        print(f"[SMTP] Falha ao enviar e-mail: {e}")
        return False

# ----------------------------------------------------------------------
# Helper para geolocalizar mercado (versão simples e precisa)
# ----------------------------------------------------------------------
def _geocode_mercado(logradouro, numero, bairro, cidade, uf, cep=None):
    logradouro = (logradouro or "").strip()
    numero     = (numero or "").strip()
    bairro     = (bairro or "").strip()
    cidade     = (cidade or "").strip()
    uf         = (uf or "").strip()
    cep        = (cep or "").strip()

    # Sem logradouro não tentamos geocodificar
    if not logradouro:
        return None, None

    partes = []

    linha1 = " ".join([logradouro, numero]).strip()
    if linha1:
        partes.append(linha1)
    if bairro:
        partes.append(bairro)
    if cidade:
        partes.append(cidade)
    if uf:
        partes.append(uf)
    if cep:
        partes.append(cep)

    endereco = ", ".join(partes)

    lat, lng = geocode_with_cache(endereco)
    if lat is None or lng is None:
        return None, None
    return lat, lng

def _parse_coord(val):
    """
    Converte texto de latitude/longitude em float.
    Aceita vírgula ou ponto. Retorna None se vazio ou inválido.
    """
    try:
        s = (val or "").strip()
        if not s:
            return None
        s = s.replace(",", ".")
        return float(s)
    except Exception:
        return None


# -----------------------------------------------------------------------------
# Token para reset de senha (itsdangerous)
# -----------------------------------------------------------------------------
def _get_serializer():
    return URLSafeTimedSerializer(app.secret_key, salt="confereapp-reset")


def gerar_token(user_id):
    s = URLSafeTimedSerializer(app.secret_key)
    return s.dumps(user_id, salt='reset-senha')


def verificar_token(token, max_age=3600):
    s = URLSafeTimedSerializer(app.secret_key)
    try:
        user_id = s.loads(token, salt='reset-senha', max_age=max_age)
    except Exception:
        return None
    return user_id


def gerar_token_reset(email: str) -> str:
    s = _get_serializer()
    return s.dumps({"email": email})


def validar_token_reset(token: str, max_age_seconds: int = 3600) -> str | None:
    s = _get_serializer()
    try:
        data = s.loads(token, max_age=max_age_seconds)
        return (data or {}).get("email")
    except (BadSignature, SignatureExpired):
        return None

# -----------------------------------------------------------------------------
# Decorators
# -----------------------------------------------------------------------------
def login_required(view):
    @wraps(view)
    def wrapped_view(*args, **kwargs):
        if 'user_id' not in session:
            # guarda a rota que a pessoa tentou acessar
            return redirect(url_for('login', next=request.path))
        return view(*args, **kwargs)
    return wrapped_view



# (novo) sistema de papéis
def admin_required(view):
    """Qualquer admin (geral ou técnico)."""
    @wraps(view)
    def wrapper(*args, **kwargs):
        uid = session.get('user_id')
        if not uid:
            return redirect(url_for('login'))
        user = db.session.get(Usuario, uid)
        if not user or not user.is_admin_any:
            abort(403)
        return view(*args, **kwargs)
    return wrapper


def admin_geral_required(view):
    """Somente Admin Geral."""
    @wraps(view)
    def wrapper(*args, **kwargs):
        uid = session.get('user_id')
        if not uid:
            return redirect(url_for('login'))
        user = db.session.get(Usuario, uid)
        if not user or not user.is_admin_geral:
            abort(403)
        return view(*args, **kwargs)
    return wrapper


def finance_access_required(view):
    """Acesso financeiro (bloqueia admin_tecnico)."""
    @wraps(view)
    def wrapper(*args, **kwargs):
        uid = session.get('user_id')
        if not uid:
            return redirect(url_for('login'))
        user = db.session.get(Usuario, uid)
        if not user or not user.is_admin_geral:
            abort(403)
        return view(*args, **kwargs)
    return wrapper


def _perfil_default_route(user):
    if user and user.is_admin_any:
        return url_for('admin_root')
    if user and user.is_empresa:
        return url_for('produtos')
    return url_for('lista')  # cliente

def cliente_or_admin_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        uid = session.get('user_id')
        if not uid:
            return redirect(url_for('login', next=request.path))
        user = db.session.get(Usuario, uid)
        if not user:
            return redirect(url_for('login'))
        if user.is_admin_any or user.is_cliente:
            return view(*args, **kwargs)
        
    return wrapper

def empresa_or_admin_required(view):
    @wraps(view)
    def wrapper(*args, **kwargs):
        uid = session.get('user_id')
        if not uid:
            return redirect(url_for('login', next=request.path))
        user = db.session.get(Usuario, uid)
        if not user:
            return redirect(url_for('login'))
        if user.is_admin_any or user.is_empresa:
            return view(*args, **kwargs)
        flash("Você não tem acesso a essa seção.", "warning")
        return redirect(_perfil_default_route(user))
    return wrapper

@app.errorhandler(403)
def forbidden(e):
    flash("Acesso não permitido para seu perfil.", "warning")
    uid = session.get('user_id')
    user = db.session.get(Usuario, uid) if uid else None
    return redirect(_perfil_default_route(user) if user else url_for('login'))

# ----------------------------------------------------------------------------- 
# Utilitários 
# -----------------------------------------------------------------------------


def _sort_key_total_then_dist(rec, total_field="total"):
    """
    Ordena por total (asc) e, em empate, por distância (asc).
    'rec' pode ser dict ou objeto com atributos.
    """
    tot = float(rec.get(total_field)) if isinstance(rec, dict) else float(getattr(rec, total_field, 0.0))
    m = rec.get("mercado") if isinstance(rec, dict) else getattr(rec, "mercado", None)
    dist = None
    if m is not None:
        dist = getattr(m, "_dist_km", None)
    dist_key = float(dist) if dist is not None else float("inf")
    return (tot, dist_key)


def allowed_file(filename: str) -> bool:
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS

def to_float(v):
    if v is None:
        return None
    if isinstance(v, (int, float)):
        return float(v)
    s = str(v).strip()
    if not s:
        return None
    s = s.replace("\xa0", "").replace("R$", "").replace(" ", "").strip()
    if "," in s and "." in s:
        s = s.replace(".", "").replace(",", ".")
    elif "," in s:
        s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None

def normalize_cnpj(cnpj: str) -> str:
    return re.sub(r"\D", "", str(cnpj or ""))

def _parse_coord(v):
    """
    Converte texto de latitude/longitude em float.
    Aceita vírgula ou ponto. Retorna None se estiver vazio ou inválido.
    """
    s = (v or "").strip()
    if not s:
        return None
    s = s.replace(",", ".")
    try:
        return float(s)
    except ValueError:
        return None


def looks_like_cnpj(cnpj: str) -> bool:
    return len(normalize_cnpj(cnpj)) == 14

# -----------------------------------------------------------------------------

# --- MIGRAÇÃO SIMPLES (SQLite): garante colunas na tabela lista_itens ---
from sqlalchemy import text

def _col_exists_sqlite(table_name, col_name):
    # usa PRAGMA table_info
    sql = text(f"PRAGMA table_info({table_name})")
    with db.engine.connect() as conn:
        rows = conn.execute(sql).fetchall()
    cols = {r[1] for r in rows}  # r[1] é o nome da coluna
    return col_name in cols

def ensure_lista_itens_columns():
    # Se a tabela não tiver as colunas, faz ALTER TABLE ADD COLUMN
    alters = []
    if not _col_exists_sqlite('lista_itens', 'marca'):
        alters.append("ALTER TABLE lista_itens ADD COLUMN marca TEXT")
    if not _col_exists_sqlite('lista_itens', 'unidade'):
        alters.append("ALTER TABLE lista_itens ADD COLUMN unidade TEXT")
    if not _col_exists_sqlite('lista_itens', 'tamanho'):
        alters.append("ALTER TABLE lista_itens ADD COLUMN tamanho REAL")

    if alters:
        with db.engine.begin() as conn:
            for sql in alters:
                conn.execute(text(sql))
        print("[OK] Migração aplicada em lista_itens:", alters)
    else:
        print("[OK] lista_itens já possui marca/unidade/tamanho.")

# Chama a migração uma vez na subida do app

# --- FIM MIGRAÇÃO ---

# -----------------------------------------------------------------------------
# Helpers de dados
# -----------------------------------------------------------------------------
def get_or_create_mercado_by_cnpj(
    nome: str,
    cnpj_raw: str,
    cidade: str = "",
    bairro: str = "",
    email: str = "",
    logradouro: str = "",
    numero: str = "",
    uf: str = "",
    cep: str = "",
    dist_km: float | None = None,
    owner_user_id: int | None = None
) -> int:
    cnpj = normalize_cnpj(cnpj_raw)
    if not cnpj:
        raise ValueError("CNPJ inválido.")

    m = Mercado.query.filter_by(cnpj=cnpj).first()
    if m:
        # se vier owner e o cadastro ainda não tem dono, grava
        if owner_user_id and not m.owner_user_id:
            m.owner_user_id = owner_user_id
            db.session.commit()
        return m.id

    m = Mercado(
        nome=nome or "Mercado sem nome",
        cnpj=cnpj,
        cidade=cidade, bairro=bairro, email=email,
        logradouro=logradouro, numero=numero, uf=uf, cep=cep,
        dist_km=dist_km,
        owner_user_id=owner_user_id
    )
    db.session.add(m)
    db.session.commit()
    return m.id


    m = Mercado(
        nome=nome, cnpj=cnpj, cidade=cidade, bairro=bairro, email=email,
        logradouro=logradouro, numero=numero, uf=uf, cep=cep, dist_km=dist_km
    )
    db.session.add(m)
    db.session.commit()
    return m.id


def cleanup_produtos_orfaos():
    subq = db.session.query(Preco.produto_id).distinct()
    q = Produto.query.filter(~Produto.id.in_(subq))
    count = q.count()
    q.delete(synchronize_session=False)
    db.session.commit()
    return count
    
    # -------------------------------------------------------------------------
# Helper: cria (ou retorna) um Produto por nome e vincula à empresa (se houver)
# -------------------------------------------------------------------------


    
# === [A] Disponibilizar o "current_user" nos templates ===
@app.context_processor
def inject_current_user():
    uid = session.get('user_id')
    user = db.session.get(Usuario, uid) if uid else None
    return {"current_user": user}

# Coloque ESTA versão única depois das classes (após class Produto)
from sqlalchemy.exc import IntegrityError
from sqlalchemy import func

def get_or_create_produto(nome: str, owner_user_id=None) -> int:
    """
    Procura produto por nome (case-insensitive). Se existir, retorna o id.
    Se não existir, cria. Evita UNIQUE com tratamento de corrida.
    """
    n = (nome or '').strip()
    if not n:
        raise ValueError("nome de produto vazio")

    # Busca case-insensitive
    q = Produto.query.filter(func.lower(Produto.nome) == n.lower())

    # Se você usa dono (empresa), permite pegar global (None) ou do dono
    if hasattr(Produto, "owner_user_id") and owner_user_id is not None:
        q = q.filter((Produto.owner_user_id == owner_user_id) | (Produto.owner_user_id.is_(None)))

    p = q.first()
    if p:
        return int(p.id)

    # Tenta criar, mas sem fazer commit (usa flush); se bater UNIQUE, reconsulta
    p = Produto(nome=n, owner_user_id=owner_user_id if hasattr(Produto, "owner_user_id") else None)
    db.session.add(p)
    try:
        db.session.flush()   # dispara o INSERT já aqui
        return int(p.id)
    except IntegrityError:
        db.session.rollback()
        p = Produto.query.filter(func.lower(Produto.nome) == n.lower()).first()
        if not p:
            raise
        return int(p.id)

# -------------------------------------------------------------------------
# Helper: cria (ou retorna) um Produto por nome (case-insensitive)
# -------------------------------------------------------------------------
from sqlalchemy import func
from sqlalchemy.exc import IntegrityError


def get_or_create_produto(nome: str, owner_user_id: int | None = None) -> int:
    """
    Localiza ou cria um Produto a partir do nome, garantindo que
    'nome_canonico' seja sempre preenchido.

    - nome: nome do produto vindo do Excel / formulário
    - owner_user_id: (opcional) id do usuário dono (se você usar isso em outro lugar)
    """

    nome_limpo = (nome or "").strip()
    if not nome_limpo:
        raise ValueError("Nome de produto vazio em get_or_create_produto().")

    # gera a versão normalizada (sem acentos, minúscula etc.)
    nome_canonico = normalize_str(nome_limpo)

    # tenta achar um produto já existente com esse nome canônico
    prod = Produto.query.filter_by(nome_canonico=nome_canonico).first()
    if prod:
        return prod.id

    # se não existir, cria um novo Produto usando o set_nome,
    # que já preenche nome e nome_canonico corretamente
    novo = Produto()
    novo.set_nome(nome_limpo)   # ESSENCIAL: preenche nome_canonico
    # se você quiser salvar categoria ou dono em algum fluxo futuro, pode usar:
    # novo.categoria = None
    # novo.owner_user_id = owner_user_id
    db.session.add(novo)
    db.session.flush()  # garante que novo.id esteja disponível

    return novo.id

def _escolhe_preco_preferindo_coincidencia(candidatos, marca=None, unidade=None, tamanho=None):
    """
    Recebe uma lista de objetos Preco e retorna o melhor candidato conforme:
      1) Preferir mesma marca (se fornecida)
      2) Preferir mesma unidade (se fornecida)
      3) Preferir tamanho mais próximo (se fornecido)
    Se nada for fornecido, retorna o menor preço.
    """
    if not candidatos:
        return None

    # Caso nenhum filtro seja informado, retorna o menor preço direto.
    if not any([marca, unidade, tamanho is not None]):
        return min(candidatos, key=lambda p: float(p.preco))

    def _score(p: 'Preco'):
        s = 0.0
        # Marca pesa mais
        if marca:
            s += 0.0 if (p.marca or '').strip().lower() == marca.strip().lower() else 10.0
        # Unidade pesa médio
        if unidade:
            s += 0.0 if (p.unidade or '').strip().lower() == unidade.strip().lower() else 5.0
        # Tamanho = distância absoluta (quanto mais perto, melhor)
        if tamanho is not None and p.tamanho is not None:
            try:
                s += abs(float(tamanho) - float(p.tamanho))
            except Exception:
                s += 2.5  # não conseguiu comparar
        elif tamanho is not None and p.tamanho is None:
            s += 1.5  # candidato sem tamanho, penaliza um pouco
        return (s, float(p.preco))  # desempate por preço

    return min(candidatos, key=_score)

def apply_fallback_if_needed(lat, lng, cidade):
    """
    Se o geocoder retornar None ou None/None,
    tenta aplicar fallback pelo centro da cidade.
    """
    if cidade:
        city_key = cidade.strip().lower()
        fallback = CITY_FALLBACK_COORDS.get(city_key)
        if fallback:
            return fallback  # lat, lng do centro da cidade
    return lat, lng


# === [B] Painel Admin resumido ===
from datetime import timedelta

@app.route("/selecionar-cidade", methods=["GET", "POST"])
@login_required
def selecionar_cidade():
    # Lista automática de cidades que têm pelo menos 1 mercado cadastrado
    cidades = (
        db.session.query(Mercado.cidade)
        .filter(Mercado.cidade.isnot(None))
        .filter(Mercado.cidade != "")
        .distinct()
        .order_by(Mercado.cidade)
        .all()
    )
    cidades = [c[0] for c in cidades]

    if request.method == "POST":
        cidade_escolhida = (request.form.get("cidade") or "").strip()
        if cidade_escolhida:
            session["cidade"] = cidade_escolhida
            flash(f"Cidade selecionada: {cidade_escolhida}", "success")
            return redirect(url_for("lista"))  # vai direto pra Minha Lista
        flash("Selecione uma cidade.", "warning")

    return render_template("selecionar_cidade.html", cidades=cidades)


@app.route('/admin/dashboard')
@login_required
@admin_required   # use @admin_geral_required se quiser restringir só a admin geral
def admin_dashboard():
    # contadores básicos
    total_usuarios = Usuario.query.count()
    ativos = Usuario.query.filter_by(ativo=True).count()
    admins_gerais = Usuario.query.filter(Usuario.role == "admin_geral", Usuario.ativo == True).count()
    admins_tecnicos = Usuario.query.filter(Usuario.role == "admin_tecnico", Usuario.ativo == True).count()

    total_mercados = Mercado.query.count()
    total_produtos = Produto.query.count()
    total_precos = Preco.query.count()

    # atividade recente (últimos 7 dias)
    limite = datetime.utcnow() - timedelta(days=7)
    precos_ultimos7 = Preco.query.filter(Preco.criado_em >= limite).count()

    # top 10 produtos com mais preços cadastrados
    top_produtos = (
        db.session.query(Produto.nome, db.func.count(Preco.id).label("n"))
        .join(Preco, Preco.produto_id == Produto.id)
        .group_by(Produto.id, Produto.nome)
        .order_by(db.desc("n"))
        .limit(10)
        .all()
    )

    # top 10 mercados com mais preços cadastrados
    top_mercados = (
        db.session.query(Mercado.nome, db.func.count(Preco.id).label("n"))
        .join(Preco, Preco.mercado_id == Mercado.id)
        .group_by(Mercado.id, Mercado.nome)
        .order_by(db.desc("n"))
        .limit(10)
        .all()
    )

    return render_template(
        'admin_dashboard.html',
        total_usuarios=total_usuarios,
        ativos=ativos,
        admins_gerais=admins_gerais,
        admins_tecnicos=admins_tecnicos,
        total_mercados=total_mercados,
        total_produtos=total_produtos,
        total_precos=total_precos,
        precos_ultimos7=precos_ultimos7,
        top_produtos=top_produtos,
        top_mercados=top_mercados
    )


# === [C] Exports CSV simples (linkados no dashboard) ===
@app.route('/admin/export/usuarios.csv')
@login_required
@admin_required
def export_usuarios_csv():
    rows = Usuario.query.order_by(Usuario.nome.asc()).all()
    output = io.StringIO()
    w = csv.writer(output, lineterminator='\n')
    w.writerow(["id","nome","email","role","ativo","criado_em"])
    for u in rows:
        w.writerow([u.id, u.nome, u.email, u.role, 1 if u.ativo else 0, u.criado_em])
    resp = Response(output.getvalue(), mimetype='text/csv; charset=utf-8')
    resp.headers['Content-Disposition'] = 'attachment; filename=usuarios.csv'
    return resp

def _period_window(period: str):
    """
    Retorna (dt_inicio, dt_fim) UTC para o período pré-definido.
    period: '7d' | '30d' | '90d' | 'all'
    """
    now = datetime.utcnow()
    if period == '7d':
        return (now - timedelta(days=7), now)
    if period == '30d':
        return (now - timedelta(days=30), now)
    if period == '90d':
        return (now - timedelta(days=90), now)
    # 'all' ou vazio -> sem filtro de data
    return (None, None)

def _localities():
    """
    Retorna lista de UFs e (UF, Cidade) existentes nos mercados.
    """
    ufs = [r[0] for r in db.session.query(Mercado.uf).filter(Mercado.uf.isnot(None), Mercado.uf!='').distinct().order_by(Mercado.uf).all()]
    cidades = db.session.query(Mercado.uf, Mercado.cidade)\
        .filter(Mercado.cidade.isnot(None), Mercado.cidade!='')\
        .distinct().order_by(Mercado.uf, Mercado.cidade).all()
    return ufs, cidades

@app.route('/admin/relatorios')
@login_required
@admin_required
def admin_relatorios():
    # filtros
    period = (request.args.get('period') or '30d')  # 7d/30d/90d/all
    uf = (request.args.get('uf') or '').strip()
    cidade = (request.args.get('cidade') or '').strip()
    produto_id = request.args.get('produto_id', type=int)

    dt_ini, dt_fim = _period_window(period)
    ufs, cidades_all = _localities()

    # ----- Consulta base de preços (para gráficos por cidade) -----
    q_precos = db.session.query(
        Preco.id, Preco.preco, Preco.criado_em,
        Produto.id.label('produto_id'), Produto.nome.label('produto_nome'),
        Mercado.cidade, Mercado.uf
    ).join(Produto, Preco.produto_id == Produto.id)\
     .join(Mercado, Preco.mercado_id == Mercado.id)

    if dt_ini and dt_fim:
        q_precos = q_precos.filter(Preco.criado_em >= dt_ini, Preco.criado_em <= dt_fim)
    if uf:
        q_precos = q_precos.filter(Mercado.uf == uf)
    if cidade:
        q_precos = q_precos.filter(Mercado.cidade == cidade)

    precos_rows = q_precos.all()

    # ----- 1) Preço médio por cidade para 1 produto -----
    if not produto_id:
        any_prod = next((r.produto_id for r in precos_rows), None)
        produto_id = any_prod

    medias_por_cidade = {}  # {cidade: média}
    produto_nome_sel = None
    if produto_id:
        valores = {}
        for r in precos_rows:
            if r.produto_id != produto_id:
                continue
            key_city = (r.cidade or 'Sem cidade').strip()
            valores.setdefault(key_city, []).append(float(r.preco))
            if produto_nome_sel is None:
                produto_nome_sel = r.produto_nome
        for c, arr in valores.items():
            if arr:
                medias_por_cidade[c] = sum(arr)/len(arr)

    # ----- 2) Produtos mais/menos adicionados à lista -----
    q_lista = db.session.query(
        Produto.nome, db.func.count(ListaItem.id).label('n')
    ).join(Produto, ListaItem.produto_id == Produto.id)

    if dt_ini and dt_fim:
        q_lista = q_lista.filter(ListaItem.criado_em >= dt_ini, ListaItem.criado_em <= dt_fim)

    q_lista = q_lista.group_by(Produto.id, Produto.nome)

    top_produtos = q_lista.order_by(db.desc('n')).limit(15).all()
    low_produtos = q_lista.order_by(db.asc('n')).limit(15).all()

    # ----- 3) Volume de preços por cidade -----
    q_vol = db.session.query(
        Mercado.cidade, db.func.count(Preco.id).label('n')
    ).join(Preco, Preco.mercado_id == Mercado.id)

    if dt_ini and dt_fim:
        q_vol = q_vol.filter(Preco.criado_em >= dt_ini, Preco.criado_em <= dt_fim)
    if uf:
        q_vol = q_vol.filter(Mercado.uf == uf)
    if cidade:
        q_vol = q_vol.filter(Mercado.cidade == cidade)

    vol_por_cidade = q_vol.group_by(Mercado.cidade).order_by(db.desc('n')).all()

    # --- VETORES PARA O TEMPLATE (evitar compreensões no Jinja) ---
    vol_labels = [(cid or 'Sem cidade') for cid, _n in vol_por_cidade]
    vol_data   = [int(_n) for _cid, _n in vol_por_cidade]

    top_labels = [nome for nome, _n in top_produtos]
    top_data   = [int(_n) for _nome, _n in top_produtos]

    low_labels = [nome for nome, _n in low_produtos]
    low_data   = [int(_n) for _nome, _n in low_produtos]

    # Dados para selects
    produtos_all = Produto.query.order_by(Produto.nome.asc()).all()

    return render_template(
        'admin_relatorios.html',
        period=period, uf=uf, cidade=cidade, produto_id=produto_id,
        ufs=ufs, cidades_all=cidades_all, produtos_all=produtos_all,
        medias_por_cidade=medias_por_cidade, produto_nome_sel=produto_nome_sel,
        top_produtos=top_produtos, low_produtos=low_produtos,
        vol_por_cidade=vol_por_cidade,
        vol_labels=vol_labels, vol_data=vol_data,
        top_labels=top_labels, top_data=top_data,
        low_labels=low_labels, low_data=low_data,
        dt_ini=dt_ini, dt_fim=dt_fim
    )

@app.route('/admin/relatorios/export/produtos_populares.csv')
@login_required
@admin_required
def rel_export_produtos_populares():
    period = (request.args.get('period') or '30d')
    dt_ini, dt_fim = _period_window(period)

    q = db.session.query(
        Produto.nome, db.func.count(ListaItem.id).label('n')
    ).join(Produto, ListaItem.produto_id == Produto.id)

    if dt_ini and dt_fim:
        q = q.filter(ListaItem.criado_em >= dt_ini, ListaItem.criado_em <= dt_fim)

    rows = q.group_by(Produto.id, Produto.nome).order_by(db.desc('n')).all()

    output = io.StringIO()
    w = csv.writer(output, lineterminator='\n')
    w.writerow(['Produto', 'Qtd_Adicoes_Lista'])
    for nome, n in rows:
        w.writerow([nome, n])
    resp = Response(output.getvalue(), mimetype='text/csv; charset=utf-8')
    resp.headers['Content-Disposition'] = 'attachment; filename=produtos_populares.csv'
    return resp


@app.route('/admin/relatorios/export/precos_por_cidade.csv')
@login_required
@admin_required
def rel_export_precos_por_cidade():
    period = (request.args.get('period') or '30d')
    uf = (request.args.get('uf') or '').strip()
    cidade = (request.args.get('cidade') or '').strip()
    produto_id = request.args.get('produto_id', type=int)

    dt_ini, dt_fim = _period_window(period)

    q = db.session.query(
        Mercado.uf, Mercado.cidade,
        Produto.nome.label('produto'),
        db.func.avg(Preco.preco).label('preco_medio'),
        db.func.count(Preco.id).label('n')
    ).join(Produto, Preco.produto_id == Produto.id)\
     .join(Mercado, Preco.mercado_id == Mercado.id)

    if dt_ini and dt_fim:
        q = q.filter(Preco.criado_em >= dt_ini, Preco.criado_em <= dt_fim)
    if uf:
        q = q.filter(Mercado.uf == uf)
    if cidade:
        q = q.filter(Mercado.cidade == cidade)
    if produto_id:
        q = q.filter(Produto.id == produto_id)

    rows = q.group_by(Mercado.uf, Mercado.cidade, Produto.nome).order_by(Mercado.uf, Mercado.cidade, Produto.nome).all()

    output = io.StringIO()
    w = csv.writer(output, lineterminator='\n')
    w.writerow(['UF','Cidade','Produto','PrecoMedio','Obs_QtdPrecos'])
    for ufv, cid, prod, pm, n in rows:
        w.writerow([ufv or '', cid or '', prod, f'{(pm or 0):.2f}', n])
    resp = Response(output.getvalue(), mimetype='text/csv; charset=utf-8')
    resp.headers['Content-Disposition'] = 'attachment; filename=precos_por_cidade.csv'
    return resp


@app.route('/admin/relatorios/export/volume_precos_por_cidade.csv')
@login_required
@admin_required
def rel_export_volume_precos_por_cidade():
    period = (request.args.get('period') or '30d')
    uf = (request.args.get('uf') or '').strip()
    cidade = (request.args.get('cidade') or '').strip()

    dt_ini, dt_fim = _period_window(period)

    q = db.session.query(
        Mercado.cidade, db.func.count(Preco.id).label('n')
    ).join(Preco, Preco.mercado_id == Mercado.id)

    if dt_ini and dt_fim:
        q = q.filter(Preco.criado_em >= dt_ini, Preco.criado_em <= dt_fim)
    if uf:
        q = q.filter(Mercado.uf == uf)
    if cidade:
        q = q.filter(Mercado.cidade == cidade)

    rows = q.group_by(Mercado.cidade).order_by(db.desc('n')).all()

    output = io.StringIO()
    w = csv.writer(output, lineterminator='\n')
    w.writerow(['Cidade','QtdPrecos'])
    for cid, n in rows:
        w.writerow([cid or 'Sem cidade', n])
    resp = Response(output.getvalue(), mimetype='text/csv; charset=utf-8')
    resp.headers['Content-Disposition'] = 'attachment; filename=volume_precos_por_cidade.csv'
    return resp


@app.route('/admin/export/mercados.csv')
@login_required
@admin_required
def export_mercados_csv():
    rows = Mercado.query.order_by(Mercado.nome.asc()).all()
    output = io.StringIO()
    w = csv.writer(output, lineterminator='\n')
    w.writerow(["id","nome","cnpj","cidade","bairro","email","lat","lng"])
    for m in rows:
        w.writerow([m.id, m.nome, m.cnpj, m.cidade, m.bairro, m.email, m.lat, m.lng])
    resp = Response(output.getvalue(), mimetype='text/csv; charset=utf-8')
    resp.headers['Content-Disposition'] = 'attachment; filename=mercados.csv'
    return resp


@app.route('/admin/export/produtos.csv')
@login_required
@admin_required
def export_produtos_csv():
    rows = Produto.query.order_by(Produto.nome.asc()).all()
    output = io.StringIO()
    w = csv.writer(output, lineterminator='\n')
    w.writerow(["id","nome"])
    for p in rows:
        w.writerow([p.id, p.nome])
    resp = Response(output.getvalue(), mimetype='text/csv; charset=utf-8')
    resp.headers['Content-Disposition'] = 'attachment; filename=produtos.csv'
    return resp


@app.route('/admin/export/precos.csv')
@login_required
@admin_required
def export_precos_csv_admin():
    rows = db.session.query(
        Produto.nome.label("produto"),
        Preco.marca, Preco.tamanho, Preco.unidade,
        Mercado.nome.label("mercado"),
        Preco.preco, Preco.criado_em
    ).join(Produto, Preco.produto_id == Produto.id)\
     .join(Mercado, Preco.mercado_id == Mercado.id)\
     .order_by(Produto.nome, Mercado.nome, Preco.criado_em.desc()).all()

    output = io.StringIO()
    w = csv.writer(output, lineterminator='\n')
    w.writerow(["Produto","Marca","Tamanho","Unidade","Mercado","Preço","Criado em"])
    for r in rows:
        w.writerow([r.produto, r.marca, r.tamanho, r.unidade, r.mercado, f"{r.preco:.2f}", r.criado_em])
    resp = Response(output.getvalue(), mimetype='text/csv; charset=utf-8')
    resp.headers['Content-Disposition'] = 'attachment; filename=precos.csv'
    return resp

# =========================================================
# CRUD de Mercados (admin + empresa)
# =========================================================

@app.route('/admin/mercados', methods=['GET', 'POST'])
@login_required
@admin_required  # apenas admin master enxerga essa tela
def admin_mercados():
    user = g.current_user
    msg = None

    if request.method == 'POST':
        acao = (request.form.get('acao') or '').strip()

        # ================= NOVO MERCADO =================
        if acao == 'novo':
            nome       = (request.form.get('nome') or '').strip()
            cnpj_raw   = (request.form.get('cnpj') or '').strip()
            email      = (request.form.get('email') or '').strip()
            logradouro = (request.form.get('logradouro') or '').strip()
            numero     = (request.form.get('numero') or '').strip()
            bairro     = (request.form.get('bairro') or '').strip()
            cidade     = (request.form.get('cidade') or '').strip()
            uf         = (request.form.get('uf') or '').strip()
            cep        = (request.form.get('cep') or '').strip()
            lat_raw    = (request.form.get('lat') or '').strip()
            lng_raw    = (request.form.get('lng') or '').strip()

            if not nome:
                msg = "⚠️ Informe o nome do mercado."
            else:
                m = Mercado(
                    nome=nome,
                    cnpj=normalize_cnpj(cnpj_raw),
                    email=email,
                    logradouro=logradouro,
                    numero=numero,
                    bairro=bairro,
                    cidade=cidade,
                    uf=uf,
                    cep=cep,
                )

                # 1º: tenta usar coordenadas manuais, se vierem preenchidas
                lat_manual = _parse_coord(lat_raw)
                lng_manual = _parse_coord(lng_raw)

                if lat_manual is not None and lng_manual is not None:
                    m.lat = lat_manual
                    m.lng = lng_manual
                else:
                    # 2º: senão, tenta geocodificar o endereço
                    lat_auto, lng_auto = _geocode_mercado(logradouro, numero, bairro, cidade, uf)
                    if lat_auto is not None and lng_auto is not None:
                        m.lat = lat_auto
                        m.lng = lng_auto

                db.session.add(m)
                db.session.commit()
                msg = "✅ Mercado cadastrado com sucesso."

        # ================= ATUALIZAR =================
        elif acao == 'atualizar':
            mercado_id = (request.form.get('id') or '').strip()
            if not mercado_id:
                msg = "⚠️ ID do mercado não informado."
            else:
                m = Mercado.query.get(mercado_id)
                if not m:
                    msg = "⚠️ Mercado não encontrado."
                else:
                    # endereço antigo (para saber se mudou)
                    old_partes = [
                        " ".join([(m.logradouro or "").strip(), (m.numero or "").strip()]).strip(),
                        (m.bairro or "").strip(),
                        (m.cidade or "").strip(),
                        (m.uf or "").strip(),
                    ]
                    old_endereco = ", ".join([p for p in old_partes if p]).strip().lower()

                    # novos dados vindos do form
                    nome       = (request.form.get('nome') or '').strip()
                    cnpj_raw   = (request.form.get('cnpj') or '').strip()
                    email      = (request.form.get('email') or '').strip()
                    logradouro = (request.form.get('logradouro') or '').strip()
                    numero     = (request.form.get('numero') or '').strip()
                    bairro     = (request.form.get('bairro') or '').strip()
                    cidade     = (request.form.get('cidade') or '').strip()
                    uf         = (request.form.get('uf') or '').strip()
                    cep        = (request.form.get('cep') or '').strip()
                    lat_raw    = (request.form.get('lat') or '').strip()
                    lng_raw    = (request.form.get('lng') or '').strip()

                    if nome:
                        m.nome = nome
                    m.cnpj       = normalize_cnpj(cnpj_raw)
                    m.email      = email
                    m.logradouro = logradouro
                    m.numero     = numero
                    m.bairro     = bairro
                    m.cidade     = cidade
                    m.uf         = uf
                    m.cep        = cep

                    # Coordenadas manuais têm prioridade
                    lat_manual = _parse_coord(lat_raw)
                    lng_manual = _parse_coord(lng_raw)

                    if lat_manual is not None and lng_manual is not None:
                        # Admin digitou coordenadas → usa exatamente essas
                        m.lat = lat_manual
                        m.lng = lng_manual
                    else:
                        # Se não veio coord. manual, mantém a lógica antiga: recalcular se endereço mudou
                        new_partes = [
                            " ".join([logradouro, numero]).strip(),
                            bairro,
                            cidade,
                            uf,
                        ]
                        new_endereco = ", ".join([p for p in new_partes if p]).strip().lower()

                        precisa_recalcular = False
                        if new_endereco != old_endereco:
                            precisa_recalcular = True
                        elif m.lat is None or m.lng is None:
                            precisa_recalcular = True

                        if precisa_recalcular:
                            lat_auto, lng_auto = _geocode_mercado(logradouro, numero, bairro, cidade, uf)
                            if lat_auto is not None and lng_auto is not None:
                                m.lat = lat_auto
                                m.lng = lng_auto
                            else:
                                # se não conseguir geocodificar o novo endereço,
                                # zera coordenadas para não ficar errado
                                m.lat = None
                                m.lng = None

                    db.session.commit()
                    msg = "✏️ Mercado atualizado."

        # ================= EXCLUIR =================
        elif acao == 'excluir':
            mercado_id = (request.form.get('id') or '').strip()
            if not mercado_id:
                msg = "⚠️ ID do mercado não informado."
            else:
                m = Mercado.query.get(mercado_id)
                if not m:
                    msg = "⚠️ Mercado não encontrado."
                else:
                    # apaga preços e desvincula usuários antes (para não dar erro de FK)
                    Preco.query.filter_by(mercado_id=m.id).delete(synchronize_session=False)
                    Usuario.query.filter_by(mercado_id=m.id).update(
                        {"mercado_id": None},
                        synchronize_session=False
                    )
                    db.session.delete(m)
                    db.session.commit()
                    msg = "🗑️ Mercado excluído com sucesso."

        # ================= DEDUPLICAR POR CNPJ =================
        elif acao == 'deduplicar':
            dups = (
                db.session.query(Mercado.cnpj, db.func.min(Mercado.id))
                .filter(Mercado.cnpj.isnot(None), Mercado.cnpj != "")
                .group_by(Mercado.cnpj)
                .having(db.func.count(Mercado.id) > 1)
                .all()
            )

            removidos = 0
            for cnpj, keep_id in dups:
                outros = Mercado.query.filter(
                    Mercado.cnpj == cnpj,
                    Mercado.id != keep_id
                ).all()
                for o in outros:
                    Preco.query.filter_by(mercado_id=o.id).update(
                        {"mercado_id": keep_id},
                        synchronize_session=False
                    )
                    Usuario.query.filter_by(mercado_id=o.id).update(
                        {"mercado_id": keep_id},
                        synchronize_session=False
                    )
                    db.session.delete(o)
                    removidos += 1

            db.session.commit()
            msg = f"🧹 Deduplicação concluída. {removidos} mercado(s) duplicado(s) removido(s)."

        else:
            msg = "⚠️ Ação inválida."

    # LISTAGEM (sempre)
    mercados = Mercado.query.order_by(Mercado.nome).all()
    return render_template('admin_mercados.html', mercados=mercados, msg=msg)


@app.route('/admin/mercados/novo', methods=['GET', 'POST'])
@login_required
@empresa_or_admin_required
def admin_mercado_novo():
    uid = session.get('user_id')
    user = db.session.get(Usuario, uid) if uid else None

    # ================================
    # POST → Cadastrar novo mercado
    # ================================
    if request.method == 'POST':
        nome        = (request.form.get('nome') or '').strip()
        cnpj_raw    = (request.form.get('cnpj') or '').strip()
        email       = (request.form.get('email') or '').strip()
        logradouro  = (request.form.get('logradouro') or '').strip()
        numero      = (request.form.get('numero') or '').strip()
        bairro      = (request.form.get('bairro') or '').strip()
        cidade      = (request.form.get('cidade') or '').strip()
        uf          = (request.form.get('uf') or '').strip()
        cep         = (request.form.get('cep') or '').strip()

        # validação simples
        if not nome:
            flash("Nome do mercado é obrigatório.", "warning")
            return render_template("admin_mercado_form.html", mercado=None)

        # normaliza CNPJ
        cnpj = normalize_cnpj(cnpj_raw)

        # cria instância do mercado
        m = Mercado(
            nome=nome,
            cnpj=cnpj,
            email=email,
            logradouro=logradouro,
            numero=numero,
            bairro=bairro,
            cidade=cidade,
            uf=uf,
            cep=cep,
        )

        # se for empresa (não admin), vincula ao mercado criado
        if user and user.is_empresa and not user.is_admin_any:
            m.owner_user_id = user.id

        # ================================
        # GEOLOCALIZAÇÃO AUTOMÁTICA
        # ================================
        lat, lng = _geocode_mercado(logradouro, numero, bairro, cidade, uf)

        if lat is not None and lng is not None:
            m.lat = lat
            m.lng = lng

        # salva no BD
        db.session.add(m)
        db.session.commit()

        flash("Mercado cadastrado com sucesso.", "success")
        return redirect(url_for('admin_mercados'))

    # ================================
    # GET → formulário vazio
    # ================================
    return render_template("admin_mercado_form.html", mercado=None)

@app.route('/admin/mercados/<int:mercado_id>/editar', methods=['GET', 'POST'])
@login_required
@empresa_or_admin_required
def admin_mercado_editar(mercado_id):
    uid = session.get('user_id')
    user = db.session.get(Usuario, uid) if uid else None
    if not user:
        return redirect(url_for('login'))

    mercado = Mercado.query.get_or_404(mercado_id)

    # empresa só pode mexer no próprio mercado
    if user.is_empresa and not user.is_admin_any:
        if user.mercado_id not in (None, mercado.id) and mercado.owner_user_id != user.id:
            abort(403)

    # endereço antigo (para comparar)
    old_partes = [
        " ".join([(mercado.logradouro or "").strip(), (mercado.numero or "").strip()]).strip(),
        (mercado.bairro or "").strip(),
        (mercado.cidade or "").strip(),
        (mercado.uf or "").strip(),
    ]
    old_endereco = ", ".join([p for p in old_partes if p]).strip().lower()

    if request.method == 'POST':
        nome        = (request.form.get('nome') or '').strip()
        cnpj_raw    = (request.form.get('cnpj') or '').strip()
        email       = (request.form.get('email') or '').strip()
        logradouro  = (request.form.get('logradouro') or '').strip()
        numero      = (request.form.get('numero') or '').strip()
        bairro      = (request.form.get('bairro') or '').strip()
        cidade      = (request.form.get('cidade') or '').strip()
        uf          = (request.form.get('uf') or '').strip()
        cep         = (request.form.get('cep') or '').strip()

        if not nome:
            flash("Nome do mercado é obrigatório.", "warning")
            return render_template('admin_mercado_form.html', mercado=mercado)

        # guarda coordenadas antigas para não perder se o geocoding falhar
        old_lat = mercado.lat
        old_lng = mercado.lng

        # atualiza dados básicos
        mercado.nome       = nome
        mercado.cnpj       = normalize_cnpj(cnpj_raw)
        mercado.email      = email
        mercado.logradouro = logradouro
        mercado.numero     = numero
        mercado.bairro     = bairro
        mercado.cidade     = cidade
        mercado.uf         = uf
        mercado.cep        = cep

        # endereço novo (string única)
        new_partes = [
            " ".join([logradouro, numero]).strip(),
            bairro,
            cidade,
            uf,
        ]
        new_endereco = ", ".join([p for p in new_partes if p]).strip().lower()

        # decide se recalcula coordenadas
        precisa_recalcular = False
        if new_endereco != old_endereco:
            # mudou alguma coisa no endereço
            precisa_recalcular = True
        elif mercado.lat is None or mercado.lng is None:
            # ainda não tinha coordenadas
            precisa_recalcular = True

        if precisa_recalcular:
            lat, lng = _geocode_mercado(logradouro, numero, bairro, cidade, uf)

            # SÓ atualiza se vier coordenada válida
            if lat is not None and lng is not None:
                mercado.lat = lat
                mercado.lng = lng
            else:
                # geocoding falhou → mantém as coordenadas antigas
                mercado.lat = old_lat
                mercado.lng = old_lng

        db.session.commit()
        flash("Mercado atualizado com sucesso.", "success")
        return redirect(url_for('admin_mercados'))

    return render_template('admin_mercado_form.html', mercado=mercado)


@app.route('/admin/mercados/excluir/<int:mercado_id>', methods=['POST'])
@login_required
@empresa_or_admin_required   # se você usa outro decorator de admin, mantenha o seu
def admin_mercado_delete(mercado_id):
    """
    Exclui um mercado de forma mais fácil:
    - apaga todos os Precos ligados a ele
    - desvincula usuários/empresas que tinham mercado_id = mercado_id
    - depois exclui o mercado em si
    """

    user = g.current_user
    if not user or not user.is_admin_any:
        abort(403)

    mercado = Mercado.query.get_or_404(mercado_id)

    # 1) Apagar todos os preços desse mercado
    qtd_precos = Preco.query.filter_by(mercado_id=mercado_id).count()
    if qtd_precos > 0:
        Preco.query.filter_by(mercado_id=mercado_id).delete(synchronize_session=False)

    # 2) Desvincular usuários (empresas) desse mercado
    usuarios_vinculados = Usuario.query.filter_by(mercado_id=mercado_id).all()
    qtd_usuarios = 0
    for u in usuarios_vinculados:
        qtd_usuarios += 1
        u.mercado_id = None
        # se quiser, transforma empresa em cliente comum
        if u.tipo == "empresa":
            u.tipo = "cliente"

    # aplica as alterações antes de deletar o mercado
    db.session.flush()

    # 3) Agora pode excluir o mercado sem violar FK
    db.session.delete(mercado)
    db.session.commit()

    # Mensagem amigável dizendo o que foi feito
    detalhes = []
    if qtd_precos:
        detalhes.append(f"{qtd_precos} preço(s) apagado(s)")
    if qtd_usuarios:
        detalhes.append(f"{qtd_usuarios} usuário(s) desvinculado(s)")

    if detalhes:
        flash(
            "🗑️ Mercado excluído com sucesso. "
            + " / ".join(detalhes)
            + ".",
            "success"
        )
    else:
        flash("🗑️ Mercado excluído com sucesso.", "success")

    return redirect(url_for('admin_mercados'))


# -----------------------------------------------------------------------------
# Rotas básicas
# -----------------------------------------------------------------------------
@app.route('/whoami')
def whoami():
    return {
        "logged_in": bool(session.get('user_id')),
        "user_id": session.get('user_id'),
        "user_nome": session.get('user_nome'),
        "user_email": session.get('user_email')
    }


@app.route("/")
def index():
    return redirect(url_for("login"))


@app.route('/admin/routes')
def admin_routes():
    return "<br>".join(sorted(str(r) for r in app.url_map.iter_rules()))

# -----------------------------------------------------------------------------
# Registro / Login / Logout
# -----------------------------------------------------------------------------
@app.route('/registrar', methods=['GET', 'POST'])
def registrar():
    if request.method == 'POST':
        nome  = (request.form.get('nome')  or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        senha = (request.form.get('senha') or '').strip()
        conf  = (request.form.get('conf')  or '').strip()

        if not (nome and email and senha and conf):
            return render_template('registrar.html', erro="Preencha todos os campos.")

        if senha != conf:
            return render_template('registrar.html', erro="A confirmação não confere com a senha.")

        if Usuario.query.filter_by(email=email).first():
            return render_template('registrar.html', erro="Já existe um usuário com esse e-mail.")

        u = Usuario(
            nome=nome,
            email=email,
            senha=generate_password_hash(senha, method='pbkdf2:sha256')
        )
        db.session.add(u)
        db.session.commit()

        session['user_id'] = u.id
        session['user_nome'] = u.nome
        session['user_email'] = u.email
        flash("✅ Conta criada. Bem-vindo(a)!", "success")
        return redirect(url_for('usuarios'))

    return render_template('registrar.html')

@app.route('/cadastro', methods=['GET'])
def cadastro():
    # página simples com dois botões: "Sou Cliente" e "Sou Empresa"
    return render_template('cadastro_tipo.html')

@app.route('/registrar_cliente', methods=['GET', 'POST'])
def registrar_cliente():
    if request.method == 'POST':
        nome  = (request.form.get('nome')  or '').strip()
        email = (request.form.get('email') or '').strip().lower()
        cpf   = (request.form.get('cpf')   or '').strip()
        senha = (request.form.get('senha') or '').strip()
        conf  = (request.form.get('conf')  or '').strip()

        if not (nome and email and cpf and senha and conf):
            return render_template('registrar_cliente.html', erro="Preencha todos os campos.")

        if senha != conf:
            return render_template('registrar_cliente.html', erro="A confirmação não confere com a senha.")

        if Usuario.query.filter_by(email=email).first():
            return render_template('registrar_cliente.html', erro="Já existe um usuário com esse e-mail.")

        u = Usuario(
            nome=nome, email=email,
            senha=generate_password_hash(senha, method='pbkdf2:sha256'),
            tipo='cliente', cpf=cpf
        )
        db.session.add(u)
        db.session.commit()

        session['user_id'] = u.id
        session['user_nome'] = u.nome
        session['user_email'] = u.email
        flash("✅ Conta de cliente criada. Bem-vindo(a)!", "success")
        return redirect(url_for('lista'))  # cliente cai direto na sua lista

    return render_template('registrar_cliente.html')

@app.route('/registrar_empresa', methods=['GET', 'POST'])
def registrar_empresa():
    if request.method == 'POST':
        emp_nome = (request.form.get('empresa_nome') or '').strip()
        cnpj     = (request.form.get('cnpj') or '').strip()
        email    = (request.form.get('email') or '').strip().lower()
        senha    = (request.form.get('senha') or '').strip()
        conf     = (request.form.get('conf') or '').strip()
        # endereço
        logradouro = (request.form.get('logradouro') or '').strip()
        numero     = (request.form.get('numero') or '').strip()
        bairro     = (request.form.get('bairro') or '').strip()
        cidade     = (request.form.get('cidade') or '').strip()
        uf         = (request.form.get('uf') or '').strip()
        cep        = (request.form.get('cep') or '').strip()

        if not (emp_nome and cnpj and email and senha and conf and cidade and uf):
            return render_template('registrar_empresa.html', erro="Preencha os campos obrigatórios (*).")

        if senha != conf:
            return render_template('registrar_empresa.html', erro="A confirmação não confere com a senha.")

        if Usuario.query.filter_by(email=email).first():
            return render_template('registrar_empresa.html', erro="Já existe um usuário com esse e-mail.")

        # cria usuário empresa
        u = Usuario(
            nome=emp_nome, email=email,
            senha=generate_password_hash(senha, method='pbkdf2:sha256'),
            tipo='empresa'
        )
        db.session.add(u); db.session.commit()

        # cria Mercado da empresa e vincula
        mid = get_or_create_mercado_by_cnpj(
            nome=emp_nome, cnpj_raw=cnpj, cidade=cidade, bairro=bairro,
            email=email, logradouro=logradouro, numero=numero, uf=uf, cep=cep
        )
        u.mercado_id = mid
        db.session.commit()

        session['user_id'] = u.id
        session['user_nome'] = u.nome
        session['user_email'] = u.email
        flash("✅ Conta de empresa criada. Bem-vindo(a)!", "success")
        return redirect(url_for('produtos'))  # empresa cai nos seus produtos

    return render_template('registrar_empresa.html')


@app.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        email = (request.form.get('email') or '').strip().lower()
        senha = (request.form.get('senha') or '').strip()

        if not email or not senha:
            return render_template('login.html', erro="Informe e-mail e senha.")

        u = Usuario.query.filter_by(email=email, ativo=True).first()
        if not u:
            return render_template('login.html', erro="E-mail ou senha inválidos.")

        senha_db = u.senha or ''
        ok = check_password_hash(senha_db, senha) if senha_db.startswith('pbkdf2:') else (senha_db == senha)
        if not ok:
            return render_template('login.html', erro="E-mail ou senha inválidos.")

        # login OK
        session.clear()
        session['user_id']    = u.id
        session['user_nome']  = u.nome
        session['user_email'] = u.email

        nxt = request.args.get('next')
        if not nxt or not nxt.startswith('/'):
            nxt = _perfil_default_route(u)
        return redirect(nxt)

    # QUALQUER requisição que NÃO seja POST cai aqui (GET, por exemplo)
    return render_template('login.html', erro=None)

@app.route('/logout')
def logout():
    session.clear()
    return redirect(url_for('login'))

# ============================ IMPORTAR EXCEL ============================ #

@app.route('/importar_excel', methods=['GET', 'POST'])
@login_required
@empresa_or_admin_required
def importar_excel():
    user = g.current_user
    is_admin = bool(user and user.is_admin_any)
    tipo = (user.tipo if user else "cliente")

    # ========== GET ==========
    if request.method == 'GET':
        # primeira abertura da tela, sem resumo ainda
        return render_template("importar_excel.html", resumo=None, erros=None)

    # ========== POST ==========
    file = request.files.get('arquivo')
    if not file or file.filename == "":
        flash("Selecione um arquivo Excel para importar.", "warning")
        return redirect(url_for('importar_excel'))

    try:
        df = pd.read_excel(file)
    except Exception as e:
        print("Erro lendo Excel:", e)
        flash("Erro ao ler arquivo Excel. Verifique o formato.", "danger")
        return redirect(url_for('importar_excel'))

    # -------- Normaliza colunas --------
    colunas = {c.strip().lower(): c for c in df.columns if isinstance(c, str)}

    def get_col(nome, obrigatoria=True):
        """
        Retorna o nome EXATO da coluna no DataFrame,
        a partir do nome normalizado (lower).
        """
        key = nome.lower()
        if key not in colunas:
            if obrigatoria:
                raise ValueError(f"Coluna obrigatória ausente: {nome}")
            return None
        return colunas[key]

    try:
        col_produto = get_col("produto")
        col_marca   = get_col("marca", obrigatoria=False)
        col_tam     = get_col("tamanho", obrigatoria=False)
        col_un      = get_col("unidade", obrigatoria=False)

        # preço pode vir como "preço" ou "preco"
        col_preco = colunas.get("preço") or colunas.get("preco")
        if not col_preco:
            raise ValueError("Coluna obrigatória ausente: preço/preco")

        # Mercado é opcional para empresa (usa mercado fixo)
        col_mercado = colunas.get("mercado")
    except ValueError as e:
        flash(str(e), "danger")
        return redirect(url_for('importar_excel'))

    # -------- 1) Mercado alvo --------
    mercado_fixo = None
    mercados_cache = {}

    if user.is_empresa and not is_admin:
        # Empresa: sempre importa para o próprio mercado
        mercado_fixo = Mercado.query.get(user.mercado_id)
        if not mercado_fixo:
            flash("Sua empresa não está vinculada a um mercado.", "danger")
            return redirect(url_for('importar_excel'))

    # -------- 2) Processar linhas --------
    importados = 0
    erros = []
    linhas_lidas = len(df.index)

    for idx, row in df.iterrows():
        try:
            # Produto
            valor_prod = row[col_produto]
            if pd.isna(valor_prod) or str(valor_prod).strip() == "":
                # linha sem produto -> ignora silenciosamente
                continue
            nome_prod = str(valor_prod).strip()

            # Marca
            marca = ""
            if col_marca:
                val = row[col_marca]
                if not pd.isna(val):
                    marca = str(val).strip()

            # Unidade
            unidade = ""
            if col_un:
                val = row[col_un]
                if not pd.isna(val):
                    unidade = str(val).strip().lower()

            # Tamanho
            tamanho = None
            if col_tam:
                val = row[col_tam]
                if not pd.isna(val) and str(val).strip() != "":
                    try:
                        tamanho = float(str(val).replace(",", "."))
                    except Exception:
                        tamanho = None  # não trava a linha só por isso

            # Preço
            preco_val = row[col_preco]
            if pd.isna(preco_val) or str(preco_val).strip() == "":
                raise ValueError(f"Linha {idx + 2}: preço vazio")

            try:
                preco = float(str(preco_val).replace(",", "."))
            except Exception:
                raise ValueError(f"Linha {idx + 2}: preço inválido -> {preco_val}")

            # -------- MERCADO --------
            if mercado_fixo:
                # Empresa: usa mercado vinculado
                mercado = mercado_fixo
            else:
                # Admin: precisa ter coluna Mercado na planilha
                if not col_mercado:
                    raise ValueError("Coluna 'Mercado' é obrigatória na planilha para administrador.")

                valor_merc = row[col_mercado]
                if pd.isna(valor_merc) or str(valor_merc).strip() == "":
                    raise ValueError(f"Linha {idx + 2}: Mercado vazio")

                nome_merc = str(valor_merc).strip()

                if nome_merc in mercados_cache:
                    mercado = mercados_cache[nome_merc]
                else:
                    mercado = Mercado.query.filter_by(nome=nome_merc).first()
                    if not mercado:
                        # cria mercado mínimo (sem endereço) só para teste
                        mercado = Mercado(nome=nome_merc)
                        db.session.add(mercado)
                        db.session.flush()
                    mercados_cache[nome_merc] = mercado

            # -------- 2.2 - Resolve o PRODUTO --------
            try:
                produto_id = get_or_create_produto(nome_prod)
            except Exception as e:
                raise ValueError(f"Erro ao criar/obter produto '{nome_prod}': {e}")

            # -------- 2.3 - Cria o PREÇO --------
            preco_obj = Preco(
                produto_id=produto_id,
                mercado_id=mercado.id,
                marca=marca,
                unidade=unidade,
                tamanho=tamanho,
                preco=preco
            )

            db.session.add(preco_obj)
            importados += 1

        except Exception as e:
            # Não derruba a importação inteira; registra erro da linha
            erros.append(str(e))
            continue

    # -------- Commit geral --------
    try:
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        erros.append(f"Erro ao gravar no banco: {e}")
        flash("Ocorreram erros ao salvar algumas informações.", "danger")

    # -------- Monta resumo --------
    resumo = (linhas_lidas, importados, len(erros))

    if importados > 0:
        flash(f"{importados} preço(s) importado(s) com sucesso.", "success")
    elif not erros:
        flash("Nenhuma linha válida encontrada no arquivo.", "warning")

    # Renderiza a própria página de importação, mostrando resumo e lista de erros
    return render_template("importar_excel.html", resumo=resumo, erros=erros)

# ========================== FIM IMPORTAR EXCEL ========================== #

# -----------------------------------------------------------------------------
# Painel/Admin de usuários (com papéis)
# -----------------------------------------------------------------------------
@app.route('/admin')
@login_required
@admin_required   # troque para @admin_geral_required se quiser só Admin Geral
def admin_root():
    return redirect(url_for('admin_dashboard'))


from sqlalchemy import or_



# -------------------------------------------------------------------
# ADMIN – GERENCIAR USUÁRIOS
# -------------------------------------------------------------------
@app.route('/admin/usuarios', methods=['GET', 'POST'])
@login_required
def admin_usuarios():
    uid = session.get('user_id')
    user_atual = db.session.get(Usuario, uid) if uid else None
    if not user_atual or not user_atual.is_admin_any:
        abort(403)

    if request.method == 'POST':
        acao    = (request.form.get('acao') or '').strip()
        user_id = int(request.form.get('user_id') or 0)

        alvo = db.session.get(Usuario, user_id)
        if not alvo:
            flash('Usuário não encontrado.', 'warning')
            return redirect(url_for('admin_usuarios'))

        is_self = (alvo.id == user_atual.id)

        if acao == 'remover':
            if is_self:
                flash('Você não pode remover a si mesmo.', 'warning')
            else:
                alvo.ativo = False
                db.session.commit()
                flash('Usuário desativado com sucesso.', 'success')

        elif acao == 'toggle_ativo':
            if is_self:
                flash('Você não pode inativar a si mesmo.', 'warning')
            else:
                alvo.ativo = not bool(alvo.ativo)
                db.session.commit()
                flash('Status do usuário atualizado.', 'success')

        elif acao == 'role':
            nova_role = (request.form.get('role') or '').strip() or 'user'
            alvo.role = nova_role
            db.session.commit()
            flash('Perfil do usuário atualizado.', 'success')

        elif acao == 'tipo':
            novo_tipo = (request.form.get('tipo') or '').strip()
            alvo.tipo = novo_tipo
            db.session.commit()
            flash('Tipo do usuário atualizado.', 'success')

        return redirect(url_for('admin_usuarios'))

    usuarios = (
        Usuario.query
        .order_by(Usuario.id.asc())
        .all()
    )

    return render_template(
        'usuarios_list.html',
        usuarios=usuarios,
        current_user_id=user_atual.id if user_atual else None
    )

# -----------------------------------------------------------------------------
# Esqueci minha senha / Reset
# -----------------------------------------------------------------------------
@app.route('/esqueci_senha', methods=['GET', 'POST'])
def esqueci_senha():
    if request.method == 'POST':
        email = request.form.get('email')
        usuario = Usuario.query.filter_by(email=email).first()
        if usuario:
            token = gerar_token(usuario.id)
            reset_url = url_for('reset_senha', token=token, _external=True)
            enviar_reset_email(email, reset_url)
        flash("Se o e-mail existir, enviamos um link para redefinir a senha.", "info")
        return redirect(url_for('login'))
    return render_template('esqueci_senha.html')

@app.route('/reset_senha/<token>', methods=['GET', 'POST'])
def reset_senha(token):
    user_id = verificar_token(token, max_age=3600)
    if not user_id:
        flash("Link inválido ou expirado. Solicite novamente.", "danger")
        return redirect(url_for('esqueci_senha'))

    usuario = db.session.get(Usuario, user_id)
    if not usuario:
        flash("Usuário não encontrado.", "danger")
        return redirect(url_for('esqueci_senha'))

    if request.method == 'POST':
        senha = (request.form.get('senha') or '').strip()
        conf  = (request.form.get('conf')  or '').strip()

        if not senha or not conf:
            flash("Preencha todos os campos.", "warning")
            return render_template('reset_senha.html', token=token, email=usuario.email)
        if senha != conf:
            flash("A confirmação não confere com a senha.", "warning")
            return render_template('reset_senha.html', token=token, email=usuario.email)
        if len(senha) < 6:
            flash("A senha deve ter pelo menos 6 caracteres.", "warning")
            return render_template('reset_senha.html', token=token, email=usuario.email)

        usuario.senha = generate_password_hash(senha, method='pbkdf2:sha256')
        db.session.commit()
        flash("Senha redefinida com sucesso. Faça login.", "success")
        return redirect(url_for('login'))

    return render_template('reset_senha.html', token=token, email=usuario.email)



# -----------------------------------------------------------------------------
# Alterar senha (políticas)
# -----------------------------------------------------------------------------
COMMON_PASSWORDS = {
    "123456","123456789","12345","qwerty","password","111111","123123","12345678",
    "abc123","1234","iloveyou","1q2w3e4r","000000","admin","admin123","senha",
    "senha123","qwertyuiop","123321","654321","user","123","brasil","102030"
}

def password_policy_errors(new_password: str, email: str = "", name: str = ""):
    errs = []
    if len(new_password) < 8:
        errs.append("A nova senha deve ter pelo menos 8 caracteres.")
    if not re.search(r"[a-z]", new_password):
        errs.append("A nova senha deve conter pelo menos 1 letra minúscula (a–z).")
    if not re.search(r"[A-Z]", new_password):
        errs.append("A nova senha deve conter pelo menos 1 letra maiúscula (A–Z).")
    if not re.search(r"[0-9]", new_password):
        errs.append("A nova senha deve conter pelo menos 1 número (0–9).")
    if not re.search(r"[^A-Za-z0-9]", new_password):
        errs.append("A nova senha deve conter pelo menos 1 símbolo (ex.: ! @ # $ % & *).")

    low = new_password.lower()
    if low in COMMON_PASSWORDS:
        errs.append("Essa senha é muito comum. Escolha outra.")
    for token in filter(None, re.split(r"[\W_]+", (email or "") + " " + (name or ""))):
        t = token.lower()
        if len(t) >= 4 and t in low:
            errs.append("A senha não deve conter seu nome ou e-mail.")
            break
    return errs

@app.route('/alterar_senha', methods=['GET', 'POST'])
@login_required
def alterar_senha():
    if request.method == 'POST':
        senha_atual = (request.form.get('senha_atual') or '').strip()
        senha_nova  = (request.form.get('senha_nova')  or '').strip()
        senha_conf  = (request.form.get('senha_conf')  or '').strip()

        u = db.session.get(Usuario, session['user_id'])
        if not u:
            flash("Usuário não encontrado.", "danger")
            return redirect(url_for('logout'))

        senha_db = u.senha or ''
        ok = check_password_hash(senha_db, senha_atual) if senha_db.startswith('pbkdf2:') else (senha_db == senha_atual)
        if not ok:
            return render_template('alterar_senha.html', erro="Senha atual incorreta.")
        if senha_nova != senha_conf:
            return render_template('alterar_senha.html', erro="A confirmação não confere com a nova senha.")
        if senha_nova == senha_atual:
            return render_template('alterar_senha.html', erro="A nova senha não pode ser igual à senha atual.")

        policy_errs = password_policy_errors(senha_nova, email=u.email or "", name=u.nome or "")
        if policy_errs:
            return render_template('alterar_senha.html', erro=" ".join(policy_errs))

        u.senha = generate_password_hash(senha_nova, method='pbkdf2:sha256')
        db.session.commit()
        flash("✅ Senha alterada com sucesso!", "success")
        return redirect(url_for('admin_usuarios'))

    return render_template('alterar_senha.html')

# -----------------------------------------------------------------------------
# Usuários (lista visível apenas a admins)
# -----------------------------------------------------------------------------
@app.route('/usuarios', endpoint='usuarios')
@login_required
@admin_required
def usuarios_redirect():
    return redirect(url_for('admin_usuarios'))

    
    # ---- Ações de administração de usuários (somente Admin Geral) ----

@app.post('/usuarios/<int:user_id>/set_role')
@login_required
@admin_geral_required
def usuarios_set_role(user_id: int):
    role = (request.form.get('role') or '').strip()
    if role not in ('user', 'admin_tecnico', 'admin_geral'):
        flash('Papel inválido.', 'warning')
        return redirect(url_for('admin_usuarios'))

    u = Usuario.query.get_or_404(user_id)

    # Se for remover admin_geral, não pode ficar sem nenhum
    if u.role == 'admin_geral' and role != 'admin_geral':
        outros_gerais = Usuario.query.filter(
            Usuario.id != u.id, Usuario.role == 'admin_geral', Usuario.ativo == True
        ).count()
        if outros_gerais == 0:
            flash('Não é possível remover o único Admin Geral.', 'warning')
            return redirect(url_for('usuarios'))

    u.role = role
    # Mantém is_admin legado coerente
    u.is_admin = True if role in ('admin_tecnico', 'admin_geral') else False
    db.session.commit()

    if role == 'admin_geral':
        flash(f'{u.nome} agora é Admin Geral.', 'success')
    elif role == 'admin_tecnico':
        flash(f'{u.nome} agora é Admin Técnico.', 'success')
    else:
        flash(f'{u.nome} agora é usuário comum.', 'success')

    return redirect(url_for('usuarios'))


@app.post('/usuarios/<int:user_id>/toggle_ativo')
@login_required
@admin_geral_required
def usuarios_toggle_ativo(user_id: int):
    u = Usuario.query.get_or_404(user_id)

    # Não permitir desativar o último admin geral
    if u.ativo and u.role == 'admin_geral':
        outros_gerais = Usuario.query.filter(
            Usuario.id != u.id, Usuario.role == 'admin_geral', Usuario.ativo == True
        ).count()
        if outros_gerais == 0:
            flash('Não é possível desativar o único Admin Geral.', 'warning')
            return redirect(url_for('admin_usuarios'))

    u.ativo = not bool(u.ativo)
    db.session.commit()
    flash(f"Usuário '{u.nome}' foi {'ativado' if u.ativo else 'desativado'}.", 'success')
    return redirect(url_for('usuarios'))


@app.post('/usuarios/<int:user_id>/delete')
@login_required
@admin_geral_required
def usuarios_delete(user_id: int):
    u = Usuario.query.get_or_404(user_id)

    # Não permitir excluir o último admin geral
    if u.role == 'admin_geral':
        outros_gerais = Usuario.query.filter(
            Usuario.id != u.id, Usuario.role == 'admin_geral', Usuario.ativo == True
        ).count()
        if outros_gerais == 0:
            flash('Não é possível excluir o único Admin Geral.', 'warning')
            return redirect(url_for('admin_usuarios'))

    db.session.delete(u)
    db.session.commit()
    flash(f"Usuário '{u.nome}' foi excluído definitivamente.", 'success')
    return redirect(url_for('usuarios'))


# -----------------------------------------------------------------------------
# Mercados (CRUD)
# -----------------------------------------------------------------------------
@app.route('/mercados', methods=['GET', 'POST'])
@login_required
@admin_required           # deixe @admin_required; troque se quiser outra regra
def mercados():
    msg = None

    if request.method == 'POST':
        acao = (request.form.get('acao') or '').strip()

        # Campos comuns do formulário
        campos = {
            "nome":       (request.form.get('nome') or '').strip(),
            "cnpj":       (request.form.get('cnpj') or '').strip(),
            "email":      (request.form.get('email') or '').strip(),
            "logradouro": (request.form.get('logradouro') or '').strip(),
            "numero":     (request.form.get('numero') or '').strip(),
            "bairro":     (request.form.get('bairro') or '').strip(),
            "cidade":     (request.form.get('cidade') or '').strip(),
            "uf":         (request.form.get('uf') or '').strip(),
            "cep":        (request.form.get('cep') or '').strip(),
        }

        if acao == 'novo':
            if not campos["nome"]:
                msg = "⚠️ Informe ao menos o nome."
            else:
                m = Mercado(**campos)

                # Se quem está criando for uma EMPRESA, grava como dona
                uid = session.get('user_id')
                user = db.session.get(Usuario, uid) if uid else None
                if user and getattr(user, "is_empresa", False):
                    m.owner_user_id = user.id

                db.session.add(m)
                db.session.commit()
                msg = "✅ Mercado adicionado."

        elif acao == 'atualizar':
            mercado_id = (request.form.get('id') or '').strip()
            if not mercado_id:
                msg = "⚠️ ID não informado."
            else:
                m = Mercado.query.get(mercado_id)
                if not m:
                    msg = "⚠️ Mercado não encontrado."
                else:
                    for k, v in campos.items():
                        setattr(m, k, v)
                    db.session.commit()
                    msg = "✏️ Mercado atualizado."

        elif acao == 'excluir':
            mercado_id = (request.form.get('id') or '').strip()
            if mercado_id:
                m = Mercado.query.get(mercado_id)
                if m:
                    db.session.delete(m)      # CASCADE remove preços vinculados
                    db.session.commit()
                    removidos = cleanup_produtos_orfaos()
                    msg = f"✅ Mercado excluído. {removidos} produto(s) órfão(s) removido(s)."
                else:
                    msg = "⚠️ Mercado não encontrado."
            else:
                msg = "⚠️ ID não informado."

        elif acao == 'deduplicar':
            dups = (
                db.session.query(Mercado.cnpj, db.func.min(Mercado.id))
                .filter(Mercado.cnpj.isnot(None), db.func.trim(Mercado.cnpj) != "")
                .group_by(Mercado.cnpj)
                .having(db.func.count(Mercado.id) > 1)
                .all()
            )
            total = 0
            for cnpj, keep_id in dups:
                outros = Mercado.query.filter(Mercado.cnpj == cnpj, Mercado.id != keep_id).all()
                for o in outros:
                    Preco.query.filter_by(mercado_id=o.id).update({"mercado_id": keep_id})
                    db.session.delete(o)
                    total += 1
            db.session.commit()
            msg = f"🧹 Deduplicação concluída. {total} duplicado(s) removido(s)."

        else:
            msg = "⚠️ Ação inválida."

    # GET (ou pós-POST): lista mercados
    mercados = Mercado.query.order_by(Mercado.nome.asc()).all()
    return render_template('mercados.html', mercados=mercados, msg=msg)

# -----------------------------------------------------------------------------
# Produtos (CRUD)
# -----------------------------------------------------------------------------
@app.route('/produtos', methods=['GET', 'POST'])
@login_required
@empresa_or_admin_required
def produtos():
    user = g.current_user
    is_admin = bool(user and user.is_admin_any)
    tipo = (user.tipo if user else "cliente")
    msg = None

    # -------------------------------------------------
    # POST: criar / atualizar / excluir / deduplicar
    # -------------------------------------------------
    if request.method == 'POST':
        acao = (request.form.get('acao') or '').strip()

        # ---------------- NOVO PRODUTO ----------------
        if acao == 'novo':
            nome = (request.form.get('nome') or '').strip()
            if not nome:
                msg = "⚠️ Informe um nome."
            else:
                existente = Produto.query.filter(Produto.nome == nome).first()
                if existente:
                    msg = "⚠️ Já existe um produto com esse nome."
                else:
                    novo = Produto(nome=nome)
                    db.session.add(novo)
                    db.session.commit()
                    msg = "✅ Produto adicionado."

        # -------------- ATUALIZAR PRODUTO --------------
        elif acao == 'atualizar':
            produto_id = request.form.get('id')
            nome = (request.form.get('nome') or '').strip()
            if not (produto_id and nome):
                msg = "⚠️ Dados insuficientes."
            else:
                p = Produto.query.get(produto_id)
                if not p:
                    msg = "⚠️ Produto não encontrado."
                else:
                    p.nome = nome
                    db.session.commit()
                    msg = "✏️ Produto atualizado."

        # --------------- EXCLUIR PRODUTO ---------------
        elif acao == 'excluir':
            produto_id = request.form.get('id')
            if not produto_id:
                msg = "⚠️ ID não informado."
            else:
                p = Produto.query.get(produto_id)
                if not p:
                    msg = "⚠️ Produto não encontrado."
                else:
                    # remove preços e itens de lista ligados a esse produto
                    Preco.query.filter_by(produto_id=p.id).delete()
                    ListaItem.query.filter_by(produto_id=p.id).delete()
                    db.session.delete(p)
                    db.session.commit()
                    msg = "🗑️ Produto excluído."

        # -------------- DEDUPLICAR PRODUTOS ------------
        elif acao == 'deduplicar':
            if not is_admin:
                msg = "⚠️ Apenas administradores podem deduplicar."
            else:
                dups = (
                    db.session.query(Produto.nome, db.func.min(Produto.id))
                    .group_by(Produto.nome)
                    .having(db.func.count(Produto.id) > 1)
                    .all()
                )
                total = 0
                for nome, keep_id in dups:
                    outros = Produto.query.filter(
                        Produto.nome == nome,
                        Produto.id != keep_id
                    ).all()
                    for o in outros:
                        Preco.query.filter_by(produto_id=o.id).update({"produto_id": keep_id})
                        ListaItem.query.filter_by(produto_id=o.id).update({"produto_id": keep_id})
                        db.session.delete(o)
                        total += 1
                db.session.commit()
                msg = f"🧹 Deduplicação concluída. {total} duplicado(s) removido(s)."

        else:
            msg = "⚠️ Ação inválida."

    # -------------------------------------------------
    # GET / Listagem
    # -------------------------------------------------
    if is_admin:
        # Admin vê todos os produtos do catálogo
        lista = Produto.query.order_by(Produto.nome).all()

    elif user and user.is_empresa:
        # Empresa vê apenas produtos que têm preços
        # cadastrados no seu mercado
        if user.mercado_id:
            lista = (
                db.session.query(Produto)
                .join(Preco, Preco.produto_id == Produto.id)
                .filter(Preco.mercado_id == user.mercado_id)
                .order_by(Produto.nome)
                .distinct()
                .all()
            )
        else:
            lista = []
    else:
        lista = []

    return render_template('produtos.html', produtos=lista, msg=msg)

# -----------------------------------------------------------------------------
# Preços
# -----------------------------------------------------------------------------
@app.route('/precos', methods=['GET', 'POST'])
@login_required
@empresa_or_admin_required
def precos():
    user = g.current_user
    is_admin = bool(user and user.is_admin_any)
    msg = None

    # -------------------------------------------------------------------------
    # 1) CADASTRO / EDIÇÃO (POST)
    # -------------------------------------------------------------------------
    if request.method == 'POST':
        produto_id = request.form.get('produto_id')
        marca      = (request.form.get('marca') or '').strip()
        tamanho    = to_float(request.form.get('tamanho'))
        unidade    = (request.form.get('unidade') or '').strip()
        preco_val  = to_float(request.form.get('preco'))

        # mercado escolhido
        if user and user.is_empresa:
            mercado_id = user.mercado_id
        else:
            mercado_id = request.form.get('mercado_id')

        if (user and user.is_empresa) and not user.mercado_id:
            msg = "⚠️ Sua conta de empresa não está vinculada a um Mercado."
        elif produto_id and mercado_id and marca and unidade and (tamanho is not None) and (preco_val is not None):
            # empresa não pode cadastrar para mercado de outro usuário
            if user and user.is_empresa and int(mercado_id) != int(user.mercado_id or 0):
                msg = "⚠️ Você só pode cadastrar preços para seu próprio Mercado."
            else:
                dup = Preco.query.filter(
                    Preco.produto_id == produto_id,
                    Preco.mercado_id == mercado_id,
                    Preco.marca == marca,
                    Preco.unidade == unidade,
                    Preco.tamanho == tamanho,
                    db.func.abs(Preco.preco - preco_val) < 1e-6
                ).first()
                if dup:
                    msg = "⚠️ Esse preço já existe para o mesmo produto/mercado!"
                else:
                    db.session.add(Preco(
                        produto_id=produto_id,
                        mercado_id=mercado_id,
                        preco=preco_val,
                        marca=marca,
                        tamanho=tamanho,
                        unidade=unidade
                    ))
                    db.session.commit()
                    msg = "✅ Preço cadastrado."
        else:
            msg = "⚠️ Preencha todos os campos corretamente."

    # -------------------------------------------------------------------------
    # 2) COMBOS (select de produtos e mercados)
    # -------------------------------------------------------------------------
    if user and user.is_empresa:
        produtos = (
            Produto.query
            .order_by(Produto.nome.asc())
            .all()
        )
        mercados = Mercado.query.filter(Mercado.id == user.mercado_id).all()
    else:
        produtos = Produto.query.order_by(Produto.nome).all()
        mercados = Mercado.query.order_by(Mercado.nome).all()

    # -------------------------------------------------------------------------
    # 3) FILTROS DA TABELA (lado direito da tela)
    # -------------------------------------------------------------------------
    f_produto = (request.args.get('f_produto') or '').strip()
    f_marca   = (request.args.get('f_marca') or '').strip()
    f_unidade = (request.args.get('f_unidade') or '').strip()

    # -------------------------------------------------------------------------
    # 4) LISTAGEM PARA A TABELA
    # -------------------------------------------------------------------------
    q = db.session.query(
        Preco.id,
        Produto.nome.label("produto"),
        Preco.marca,
        Preco.tamanho,
        Preco.unidade,
        Mercado.nome.label("mercado"),
        Preco.preco,
        Preco.criado_em
    ).join(Produto, Preco.produto_id == Produto.id) \
     .join(Mercado, Preco.mercado_id == Mercado.id)

    # Visibilidade por tipo de usuário
    if user and getattr(user, "is_empresa", False) and getattr(user, "mercado_id", None):
        # Empresa: só vê o próprio mercado
        q = q.filter(Preco.mercado_id == user.mercado_id)
    elif user and getattr(user, "is_admin_any", False):
        # Admin geral: vê todos os mercados (não filtra por mercado)
        pass
    else:
        # (caso alguém caia aqui sem ser empresa/admin, não retorna nada)
        q = q.filter(False)

    # Aplicar filtros de busca se o usuário preencheu
    if f_produto:
        q = q.filter(Produto.nome.ilike(f"%{f_produto}%"))
    if f_marca:
        q = q.filter(Preco.marca.ilike(f"%{f_marca}%"))
    if f_unidade:
        q = q.filter(Preco.unidade.ilike(f"%{f_unidade}%"))

    lista = q.order_by(Preco.id.desc()).limit(200).all()

    # -------------------------------------------------------------------------
    # 5) RENDER
    # -------------------------------------------------------------------------
    return render_template(
        'precos.html',
        produtos=produtos,
        mercados=mercados,
        marcas_sugestoes=[r[0] for r in db.session.query(Preco.marca).distinct().order_by(Preco.marca).all()],
        unidades_sugestoes=[r[0] for r in db.session.query(Preco.unidade).distinct().order_by(Preco.unidade).all()],
        tamanhos_sugestoes=[r[0] for r in db.session.query(Preco.tamanho).distinct().order_by(Preco.tamanho).all()],
        lista=lista,
        msg=msg,
        f_produto=f_produto,
        f_marca=f_marca,
        f_unidade=f_unidade,
    )

from flask import render_template

# Função auxiliar: aqui será a "fonte da verdade" da lista de compras
def _carregar_itens_minha_lista(usuario):
    """
    POR ENQUANTO: retorna lista vazia (nenhum item).
    Depois você vai trocar esse trecho pela consulta real ao banco.
    """
    itens = []   # <-- TROCAR por uma query real depois
    total = 0.0  # soma dos subtotais

    # EXEMPLO de como deve ficar no futuro:
    # itens = MinhaListaItem.query.filter_by(usuario_id=usuario.id).all()
    # total = sum((i.preco or 0) * (i.quantidade or 1) for i in itens)

    return itens, total



from flask import Response
import csv
from io import StringIO


@app.route('/preco/excluir/<int:preco_id>', methods=['POST'])
@login_required
def excluir_preco(preco_id):
    p = Preco.query.get(preco_id)
    if p:
        db.session.delete(p)
        db.session.commit()
        flash("🗑️ Preço excluído.")
    return redirect(url_for('precos'))

# -----------------------------------------------------------------------------
# Lista do usuário
# -----------------------------------------------------------------------------
# imports no topo se ainda não tiver:
from datetime import datetime
from flask import flash

# ------------------------------------------------------------
# Remover item da lista (por id)  - SOMENTE POST
# ------------------------------------------------------------
@app.route('/lista/remover/<int:item_id>', methods=['POST'])
@login_required
def remover_item(item_id):
    try:
        item = (
            ListaItem
            .query
            .filter_by(id=item_id, user_id=session['user_id'])
            .first()
        )

        if not item:
            flash('Item não encontrado.')
            return redirect(url_for('lista'))

        db.session.delete(item)
        db.session.commit()
        flash('Item removido com sucesso.')
    except Exception as e:
        db.session.rollback()
        flash(f'Erro ao remover item: {e}')

    return redirect(url_for('lista'))


# ------------------------------------------------------------
# Minha Lista (GET: exibe / POST: adiciona item)
# ------------------------------------------------------------
@app.route('/lista', methods=['GET', 'POST'])
@login_required
def lista():
    uid = session['user_id']
    msg = None

    # ------------------------------------
    # VERIFICA SE CIDADE ESTÁ SELECIONADA
    # ------------------------------------
    cidade = session.get("cidade")
    if not cidade:
        flash("Selecione a cidade onde você está comprando.", "warning")
        return redirect(url_for("selecionar_cidade"))

    # ------------------------------
    # 1) POST: adicionar item à lista
    # ------------------------------
    if request.method == 'POST':

        produto_id  = (request.form.get('produto_id') or '').strip()
        qtd_str     = (request.form.get('qtd') or '1').strip()
        tamanho_str = (request.form.get('tamanho') or '').strip()

        if not produto_id:
            flash("Selecione um produto.", "warning")
            return redirect(url_for('lista'))

        try:
            quantidade = int(qtd_str)
            if quantidade < 1:
                raise ValueError
        except ValueError:
            quantidade = 1

        if tamanho_str:
            try:
                tamanho_val = float(tamanho_str.replace(',', '.'))
            except ValueError:
                tamanho_val = None
        else:
            tamanho_val = None

        novo_item = ListaItem(
            user_id=uid,
            produto_id=int(produto_id),
            quantidade=quantidade,
            marca=None,
            unidade=None,
            tamanho=tamanho_val
        )

        db.session.add(novo_item)
        db.session.commit()
        flash("Item adicionado à lista.", "success")
        return redirect(url_for('lista'))

    # -----------------------------------
    # 2) GET (e também após qualquer POST)
    # -----------------------------------

    # 🧠 AJUSTE PRINCIPAL:
    # Buscamos apenas produtos com preço em mercados da cidade selecionada
    rows_prod = (
        db.session.query(Produto.id, Produto.nome)
        .join(Preco, Preco.produto_id == Produto.id)
        .join(Mercado, Mercado.id == Preco.mercado_id)
        .filter(Mercado.cidade == cidade)
        .group_by(Produto.id, Produto.nome)
        .order_by(Produto.nome)
        .all()
    )

    canon_to_rep_id = {}
    rep_products    = []

    for pid, nome in rows_prod:
        canon = _canon_name(nome)
        if not canon:
            continue
        if canon not in canon_to_rep_id:
            canon_to_rep_id[canon] = pid
            rep_products.append(SimpleNamespace(id=pid, nome=nome))

    # 2.2 Itens do usuário
    itens = (
        db.session.query(
            ListaItem.id.label('item_id'),
            Produto.nome.label('produto'),
            ListaItem.marca.label('marca'),
            ListaItem.unidade.label('unidade'),
            ListaItem.tamanho.label('tamanho'),
            ListaItem.quantidade.label('quantidade'),
        )
        .join(Produto, Produto.id == ListaItem.produto_id)
        .filter(ListaItem.user_id == uid)
        .order_by(ListaItem.id.desc())
        .all()
    )

    # 2.3 Variações (marca, unidade, tamanho)
    rows_var = (
        db.session.query(
            Produto.id.label('produto_id'),
            Produto.nome.label('produto_nome'),
            Preco.marca.label('marca'),
            Preco.unidade.label('unidade'),
            Preco.tamanho.label('tamanho'),
        )
        .join(Produto, Preco.produto_id == Produto.id)
        .join(Mercado, Mercado.id == Preco.mercado_id)
        .filter(Mercado.cidade == cidade)
        .all()
    )

    tmp_sets = defaultdict(lambda: {
        "marcas": set(),
        "unidades": set(),
        "tamanhos": set(),
    })

    for r in rows_var:
        canon = _canon_name(r.produto_nome)
        rep_id = canon_to_rep_id.get(canon)
        if not rep_id:
            continue

        m = (r.marca or '').strip()
        u = (r.unidade or '').strip().lower()

        try:
            t = float(r.tamanho or 0.0)
        except:
            t = 0.0

        if m:
            tmp_sets[rep_id]["marcas"].add(m)
        if u:
            tmp_sets[rep_id]["unidades"].add(u)
        if t and t != 0.0:
            tmp_sets[rep_id]["tamanhos"].add(t)

    produto_variantes = {}
    for rep_id, sets in tmp_sets.items():
        unidades = sorted(sets["unidades"])
        tamanhos = sorted(sets["tamanhos"])
        produto_variantes[rep_id] = {
            "unidades": unidades,
            "tamanhos": tamanhos,
        }

    return render_template(
        'lista.html',
        produtos=rep_products,
        itens=itens,
        produto_variantes=produto_variantes,
        msg=msg,
        cidade=cidade,
    )


# ====== [ROTA /comparar — SUBSTITUA A SUA ATUAL POR ESTA] ======

def _build_comparar_context(lat, lng, raio_km, uid):
    """
    Monta todo o contexto usado na comparação de preços.
    Essa função é basicamente a lógica da rota /comparar,
    mas em vez de render_template ela devolve um dict.
    """

    mercados_all = Mercado.query.all()
    mercados = []

    if lat is not None and lng is not None:
        # Filtro por "caixa" (bounding box)
        lat_deg = raio_km / 111.0
        lng_deg = raio_km / max(1e-9, (111.0 * cos(radians(lat))))

        cand = [
            m for m in mercados_all
            if (m.lat is not None and m.lng is not None and
                abs(m.lat - lat) <= lat_deg and abs(m.lng - lng) <= lng_deg)
        ]

        # Distância real (Haversine) + corte por raio
        for m in cand:
            d = haversine_km(lat, lng, m.lat, m.lng)
            if d is not None and d <= raio_km:
                m._dist_km = round(d, 2)
                mercados.append(m)

        # Ordena por distância
        mercados.sort(key=lambda mm: mm._dist_km)
    else:
        # Sem localização → usa todos, sem distância
        for m in mercados_all:
            m._dist_km = None
        mercados = list(mercados_all)

    aviso = None

    # 2) Sem mercados
    if not mercados:
        aviso = ("Nenhum mercado encontrado no raio selecionado."
                 if (lat is not None and lng is not None)
                 else "Informe sua localização ou ajuste o raio.")
        return dict(
            mercados=[],
            best_single_total=None,
            best_single_mercado=None,
            best_single_map={},
            best_split_total=None,
            best_split_map={},
            best_split_groups=[],
            economia=None,
            lat=lat, lng=lng, raio_km=raio_km,
            aviso=aviso,
            nomes_produtos={}
        )

    mercado_ids = [int(m.id) for m in mercados]

    # 3) Itens da lista do usuário
    itens_rows = (
        db.session.query(
            Produto.id,
            Produto.nome,
            func.coalesce(ListaItem.quantidade, 1.0)
        )
        .join(ListaItem, ListaItem.produto_id == Produto.id)
        .filter(ListaItem.user_id == uid)
        .all()
    )

    if not itens_rows:
        aviso = "Sua lista está vazia. Adicione itens para comparar."
        return dict(
            mercados=mercados,
            best_single_total=None,
            best_single_mercado=None,
            best_single_map={},
            best_split_total=None,
            best_split_map={},
            best_split_groups=[],
            economia=None,
            lat=lat, lng=lng, raio_km=raio_km,
            aviso=aviso,
            nomes_produtos={}
        )

    produto_ids = [int(pid) for (pid, _nome, _q) in itens_rows]
    nomes_produtos = {int(pid): nome for (pid, nome, _q) in itens_rows}
    itens_lista = [(int(pid), float(qtd)) for (pid, _nome, qtd) in itens_rows]

    num_itens_lista = len(itens_lista)

    # 4) Ofertas: menor preço por (produto, mercado)
    best_variant = _build_offers_for_items(produto_ids, mercado_ids)

    if not best_variant:
        aviso = "Não há preços cadastrados para os itens desta lista nos mercados selecionados."
        return dict(
            mercados=mercados,
            best_single_total=None,
            best_single_mercado=None,
            best_single_map={},
            best_split_total=None,
            best_split_map={},
            best_split_groups=[],
            economia=None,
            lat=lat, lng=lng, raio_km=raio_km,
            aviso=aviso,
            nomes_produtos=nomes_produtos
        )

    # Distâncias por mercado
    dist_map = {int(m.id): _dist_km_or_zero(lat, lng, m) for m in mercados}

    # 5) Melhor mercado único
    single_alloc, single_used, single_total = _solve_allocation_ilp(
        itens_lista,
        mercados,
        best_variant,
        dist_map,
        max_markets=1,
        lambda_per_market=0.0,
        alpha_per_km=0.0
    )

    best_single_total = single_total if single_alloc else None
    best_single_mercado = None
    best_single_map = {}

    if single_alloc and single_used:
        mid = next(iter(single_used))
        best_single_mercado = next(
            (m for m in mercados if int(m.id) == int(mid)), None
        )
        best_single_map = {pid: info for pid, info in single_alloc.items()}

    # 6) Top 3 fracionado com regra de cobertura
    cobertura_por_mercado = defaultdict(int)
    for (pid, mid), info in best_variant.items():
        cobertura_por_mercado[int(mid)] += 1

    MIN_FRAC_COBERTURA = 0.30
    min_itens_por_mercado = max(
        1,
        int(ceil(MIN_FRAC_COBERTURA * num_itens_lista))
    )

    mercados_candidatos = []
    for m in mercados:
        mid = int(m.id)
        cob = cobertura_por_mercado.get(mid, 0)
        if cob >= min_itens_por_mercado:
            mercados_candidatos.append((m, cob))

    if not mercados_candidatos:
        mercados_split = list(mercados)
    else:
        mercados_candidatos.sort(
            key=lambda mc: (-mc[1], (mc[0].__dict__.get("_dist_km") or 0.0))
        )
        mercados_split = [m for (m, _cob) in mercados_candidatos]

    if len(mercados_split) > 3:
        mercados_split = mercados_split[:3]

    dist_map_split = {int(m.id): dist_map[int(m.id)] for m in mercados_split}

    LAMBDA_MARKET = 0.0
    ALPHA_PER_KM  = 0.0

    split_alloc, split_used, split_total = _solve_allocation_ilp(
        itens_lista,
        mercados_split,
        best_variant,
        dist_map_split,
        max_markets=min(3, len(mercados_split)),
        lambda_per_market=LAMBDA_MARKET,
        alpha_per_km=ALPHA_PER_KM
    )

    best_split_total = split_total if split_alloc else None
    best_split_map = {pid: info for pid, info in (split_alloc or {}).items()}

    # 7) Agrupar por mercado para exibir na UI
    groups_dict = defaultdict(
        lambda: {"mercado": None, "itens": [], "subtotal": 0.0, "qtd_itens": 0}
    )

    for pid, info in best_split_map.items():
        mid = int(info["mercado_id"])
        mm = next((m for m in mercados_split if int(m.id) == mid), None)
        if not mm:
            continue

        g = groups_dict[mid]
        if g["mercado"] is None:
            g["mercado"] = mm

        qtd   = float(info["qtd"])
        preco = float(info["preco"])
        linha_total = qtd * preco

        g["itens"].append({
            "pid": pid,
            "nome": nomes_produtos.get(pid, f"#{pid}"),
            "qtd": qtd,
            "preco": preco,
            "total_linha": linha_total,
            "marca": info.get("marca") or "",
            "unidade": info.get("unidade") or "",
            "tamanho": float(info.get("tamanho") or 0.0),
        })
        g["subtotal"]   += linha_total
        g["qtd_itens"]  += 1

    best_split_groups = sorted(
        groups_dict.values(),
        key=lambda G: (
            G["subtotal"],
            (G["mercado"].__dict__.get("_dist_km") or 0.0)
        )
    )

    # 8) Economia
    economia = None
    if best_single_total is not None and best_split_total is not None:
        economia = max(0.0, best_single_total - best_split_total)

    # 9) Retorna contexto
    return dict(
        mercados=mercados,
        best_single_total=best_single_total,
        best_single_mercado=best_single_mercado,
        best_single_map=best_single_map,
        best_split_total=best_split_total,
        best_split_map=best_split_map,
        best_split_groups=best_split_groups,
        economia=economia,
        lat=lat, lng=lng, raio_km=raio_km,
        aviso=aviso,
        nomes_produtos=nomes_produtos
    )

from io import BytesIO
from datetime import datetime


from flask import send_file

def _calcular_comparacao(lat, lng, raio_km):
    """
    Faz todos os cálculos da comparação e devolve um dicionário com os dados
    para usar tanto no HTML quanto no PDF.
    """
    mercados_all = Mercado.query.all()
    mercados = []

    if lat is not None and lng is not None:
        # Filtro rápido por "caixa" (bounding box)
        lat_deg = raio_km / 111.0
        lng_deg = raio_km / max(1e-9, (111.0 * cos(radians(lat))))

        cand = [
            m for m in mercados_all
            if (m.lat is not None and m.lng is not None and
                abs(m.lat - lat) <= lat_deg and abs(m.lng - lng) <= lng_deg)
        ]

        # Distância real (Haversine) e corte por raio
        for m in cand:
            d = haversine_km(lat, lng, m.lat, m.lng)
            if d is not None and d <= raio_km:
                m._dist_km = round(d, 2)
                mercados.append(m)

        # Ordena por distância crescente
        mercados.sort(key=lambda mm: mm._dist_km)
    else:
        # Sem localização → usa todos, sem distância
        for m in mercados_all:
            m._dist_km = None
        mercados = list(mercados_all)

    aviso = None

    # 2) Sem mercados
    if not mercados:
        aviso = ("Nenhum mercado encontrado no raio selecionado."
                 if (lat is not None and lng is not None)
                 else "Informe sua localização ou ajuste o raio.")
        return dict(
            mercados=[],
            best_single_total=None,
            best_single_mercado=None,
            best_single_map={},
            best_split_total=None,
            best_split_map={},
            best_split_groups=[],
            economia=None,
            aviso=aviso,
            nomes_produtos={},
        )

    mercado_ids = [int(m.id) for m in mercados]

    # 3) Itens da lista do usuário
    itens_rows = (
        db.session.query(
            Produto.id,
            Produto.nome,
            func.coalesce(ListaItem.quantidade, 1.0)
        )
        .join(ListaItem, ListaItem.produto_id == Produto.id)
        .filter(ListaItem.user_id == session['user_id'])
        .all()
    )

    if not itens_rows:
        aviso = "Sua lista está vazia. Adicione itens para comparar."
        return dict(
            mercados=mercados,
            best_single_total=None,
            best_single_mercado=None,
            best_single_map={},
            best_split_total=None,
            best_split_map={},
            best_split_groups=[],
            economia=None,
            aviso=aviso,
            nomes_produtos={},
        )

    produto_ids = [int(pid) for (pid, _nome, _q) in itens_rows]
    nomes_produtos = {int(pid): nome for (pid, nome, _q) in itens_rows}
    itens_lista = [(int(pid), float(qtd)) for (pid, _nome, qtd) in itens_rows]

    num_itens_lista = len(itens_lista)

    # 4) Ofertas (menor preço por produto/mercado)
    best_variant = _build_offers_for_items(produto_ids, mercado_ids)

    if not best_variant:
        aviso = "Não há preços cadastrados para os itens desta lista nos mercados selecionados."
        return dict(
            mercados=mercados,
            best_single_total=None,
            best_single_mercado=None,
            best_single_map={},
            best_split_total=None,
            best_split_map={},
            best_split_groups=[],
            economia=None,
            aviso=aviso,
            nomes_produtos=nomes_produtos,
        )

    # Distâncias por mercado
    dist_map = {int(m.id): _dist_km_or_zero(lat, lng, m) for m in mercados}

    # 5) Melhor mercado único
    single_alloc, single_used, single_total = _solve_allocation_ilp(
        itens_lista,
        mercados,
        best_variant,
        dist_map,
        max_markets=1,
        lambda_per_market=0.0,
        alpha_per_km=0.0
    )

    best_single_total = single_total if single_alloc else None
    best_single_mercado = None
    best_single_map = {}

    if single_alloc and single_used:
        mid = next(iter(single_used))
        best_single_mercado = next(
            (m for m in mercados if int(m.id) == int(mid)), None
        )
        best_single_map = {pid: info for pid, info in single_alloc.items()}

    # 6) Regra de cobertura para Top 3 fracionado
    cobertura_por_mercado = defaultdict(int)
    for (pid, mid), info in best_variant.items():
        cobertura_por_mercado[int(mid)] += 1

    MIN_FRAC_COBERTURA = 0.30
    min_itens_por_mercado = max(
        1,
        int(ceil(MIN_FRAC_COBERTURA * num_itens_lista))
    )

    mercados_candidatos = []
    for m in mercados:
        mid = int(m.id)
        cob = cobertura_por_mercado.get(mid, 0)
        if cob >= min_itens_por_mercado:
            mercados_candidatos.append((m, cob))

    if not mercados_candidatos:
        mercados_split = list(mercados)
    else:
        mercados_candidatos.sort(
            key=lambda mc: (-mc[1], (mc[0].__dict__.get("_dist_km") or 0.0))
        )
        mercados_split = [m for (m, _cob) in mercados_candidatos]

    if len(mercados_split) > 3:
        mercados_split = mercados_split[:3]

    dist_map_split = {int(m.id): dist_map[int(m.id)] for m in mercados_split}

    LAMBDA_MARKET = 0.0
    ALPHA_PER_KM  = 0.0

    split_alloc, split_used, split_total = _solve_allocation_ilp(
        itens_lista,
        mercados_split,
        best_variant,
        dist_map_split,
        max_markets=min(3, len(mercados_split)),
        lambda_per_market=LAMBDA_MARKET,
        alpha_per_km=ALPHA_PER_KM
    )

    best_split_total = split_total if split_alloc else None
    best_split_map = {pid: info for pid, info in (split_alloc or {}).items()}

    # 7) Agrupar por mercado (Top 3 fracionado)
    groups_dict = defaultdict(
        lambda: {"mercado": None, "itens": [], "subtotal": 0.0, "qtd_itens": 0}
    )

    for pid, info in best_split_map.items():
        mid = int(info["mercado_id"])
        mm = next((m for m in mercados_split if int(m.id) == mid), None)
        if not mm:
            continue

        g = groups_dict[mid]
        if g["mercado"] is None:
            g["mercado"] = mm

        qtd   = float(info["qtd"])
        preco = float(info["preco"])
        linha_total = qtd * preco

        g["itens"].append({
            "pid": pid,
            "nome": nomes_produtos.get(pid, f"#{pid}"),
            "qtd": qtd,
            "preco": preco,
            "total_linha": linha_total,
            "marca": info.get("marca") or "",
            "unidade": info.get("unidade") or "",
            "tamanho": float(info.get("tamanho") or 0.0),
        })
        g["subtotal"]   += linha_total
        g["qtd_itens"]  += 1

    best_split_groups = sorted(
        groups_dict.values(),
        key=lambda G: (
            G["subtotal"],
            (G["mercado"].__dict__.get("_dist_km") or 0.0)
        )
    )

    # 8) Economia
    economia = None
    if best_single_total is not None and best_split_total is not None:
        economia = max(0.0, best_single_total - best_split_total)

    return dict(
        mercados=mercados,
        best_single_total=best_single_total,
        best_single_mercado=best_single_mercado,
        best_single_map=best_single_map,
        best_split_total=best_split_total,
        best_split_map=best_split_map,
        best_split_groups=best_split_groups,
        economia=economia,
        aviso=aviso,
        nomes_produtos=nomes_produtos,
    )


@app.route('/comparar', methods=['GET'])
@login_required
def comparar(make_internal_call=False, _lat=None, _lng=None, _raio_km=None):
    # ------------------------------------------------------------------
    # 1) Parâmetros de localização
    # ------------------------------------------------------------------
    if make_internal_call:
        # chamada interna (PDF)
        lat = _lat
        lng = _lng
        raio_km = _raio_km or 10.0
    else:
        # chamada normal via navegador
        lat = _to_float(request.args.get('lat'))
        lng = _to_float(request.args.get('lng'))
        raio_km = _to_float(request.args.get('raio_km')) or 10.0

    mercados_all = Mercado.query.all()
    mercados = []

    if lat is not None and lng is not None:
        lat_deg = raio_km / 111.0
        from math import cos, radians
        lng_deg = raio_km / max(1e-9, (111.0 * cos(radians(lat))))

        cand = [
            m for m in mercados_all
            if (m.lat is not None and m.lng is not None and
                abs(m.lat - lat) <= lat_deg and abs(m.lng - lng) <= lng_deg)
        ]

        for m in cand:
            d = haversine_km(lat, lng, m.lat, m.lng)
            if d is not None and d <= raio_km:
                m._dist_km = round(d, 2)
                mercados.append(m)

        mercados.sort(key=lambda mm: mm._dist_km)
    else:
        for m in mercados_all:
            m._dist_km = None
        mercados = list(mercados_all)

    aviso = None
    economia = None
    economia_modo = None  # 'single', 'split' ou 'igual'

    # ------------------------------------------------------------------
    # 2) Se não há mercados válidos
    # ------------------------------------------------------------------
    if not mercados:
        aviso = ("Nenhum mercado encontrado no raio selecionado."
                 if (lat is not None and lng is not None)
                 else "Informe sua localização ou ajuste o raio.")

        contexto_vazio = {
            "mercados": [],
            "best_single_total": None,
            "best_single_mercado": None,
            "best_single_map": {},
            "best_split_total": None,
            "best_split_map": {},
            "best_split_groups": [],
            "economia": None,
            "economia_modo": None,
            "lat": lat, "lng": lng, "raio_km": raio_km,
            "aviso": aviso,
            "nomes_produtos": {},
        }

        if make_internal_call:
            return contexto_vazio

        return render_template('comparar.html', **contexto_vazio)

    mercado_ids = [int(m.id) for m in mercados]

    # ------------------------------------------------------------------
    # 3) Itens da lista do usuário
    # ------------------------------------------------------------------
    itens_rows = (
        db.session.query(
            Produto.id,
            Produto.nome,
            func.coalesce(ListaItem.quantidade, 1.0)
        )
        .join(ListaItem, ListaItem.produto_id == Produto.id)
        .filter(ListaItem.user_id == session['user_id'])
        .all()
    )

    if not itens_rows:
        aviso = "Sua lista está vazia. Adicione itens para comparar."

        contexto_vazio = {
            "mercados": mercados,
            "best_single_total": None,
            "best_single_mercado": None,
            "best_single_map": {},
            "best_split_total": None,
            "best_split_map": {},
            "best_split_groups": [],
            "economia": None,
            "economia_modo": None,
            "lat": lat, "lng": lng, "raio_km": raio_km,
            "aviso": aviso,
            "nomes_produtos": {},
        }

        if make_internal_call:
            return contexto_vazio

        return render_template('comparar.html', **contexto_vazio)

    produto_ids = [int(pid) for (pid, _nome, _q) in itens_rows]
    nomes_produtos = {int(pid): nome for (pid, nome, _q) in itens_rows}
    itens_lista = [(int(pid), float(qtd)) for (pid, _nome, qtd) in itens_rows]
    num_itens_lista = len(itens_lista)

    # ------------------------------------------------------------------
    # 4) Ofertas (melhor preço por produto/mercado)
    # ------------------------------------------------------------------
    best_variant = _build_offers_for_items(produto_ids, mercado_ids)

    if not best_variant:
        aviso = "Não há preços cadastrados para os itens desta lista nos mercados selecionados."

        contexto_vazio = {
            "mercados": mercados,
            "best_single_total": None,
            "best_single_mercado": None,
            "best_single_map": {},
            "best_split_total": None,
            "best_split_map": {},
            "best_split_groups": [],
            "economia": None,
            "economia_modo": None,
            "lat": lat, "lng": lng, "raio_km": raio_km,
            "aviso": aviso,
            "nomes_produtos": nomes_produtos,
        }

        if make_internal_call:
            return contexto_vazio

        return render_template('comparar.html', **contexto_vazio)

    dist_map = {int(m.id): _dist_km_or_zero(lat, lng, m) for m in mercados}

    # ------------------------------------------------------------------
    # 5) Melhor mercado ÚNICO
    # ------------------------------------------------------------------
    single_alloc, single_used, single_total = _solve_allocation_ilp(
        itens_lista,
        mercados,
        best_variant,
        dist_map,
        max_markets=1,
        lambda_per_market=0.0,
        alpha_per_km=0.0
    )

    best_single_total = single_total if single_alloc else None
    best_single_mercado = None
    best_single_map = {}

    if single_alloc and single_used:
        mid = next(iter(single_used))
        best_single_mercado = next(
            (m for m in mercados if int(m.id) == int(mid)), None
        )
        best_single_map = {pid: info for pid, info in single_alloc.items()}

    # ------------------------------------------------------------------
    # 6) TOP 3 FRACIONADO com regra de cobertura
    # ------------------------------------------------------------------
    cobertura_por_mercado = defaultdict(int)
    for (pid, mid), info in best_variant.items():
        cobertura_por_mercado[int(mid)] += 1

    MIN_FRAC_COBERTURA = 0.30
    from math import ceil
    min_itens_por_mercado = max(
        1,
        int(ceil(MIN_FRAC_COBERTURA * num_itens_lista))
    )

    mercados_candidatos = []
    for m in mercados:
        mid = int(m.id)
        cob = cobertura_por_mercado.get(mid, 0)
        if cob >= min_itens_por_mercado:
            mercados_candidatos.append((m, cob))

    if not mercados_candidatos:
        mercados_split = list(mercados)
    else:
        mercados_candidatos.sort(
            key=lambda mc: (-mc[1], (mc[0].__dict__.get("_dist_km") or 0.0))
        )
        mercados_split = [m for (m, _cob) in mercados_candidatos]

    if len(mercados_split) > 3:
        mercados_split = mercados_split[:3]

    dist_map_split = {int(m.id): dist_map[int(m.id)] for m in mercados_split}

    LAMBDA_MARKET = 0.0
    ALPHA_PER_KM  = 0.0

    split_alloc, split_used, split_total = _solve_allocation_ilp(
        itens_lista,
        mercados_split,
        best_variant,
        dist_map_split,
        max_markets=min(3, len(mercados_split)),
        lambda_per_market=LAMBDA_MARKET,
        alpha_per_km=ALPHA_PER_KM
    )

    best_split_total = split_total if split_alloc else None
    best_split_map = {pid: info for pid, info in (split_alloc or {}).items()}

    groups_dict = defaultdict(
        lambda: {"mercado": None, "itens": [], "subtotal": 0.0, "qtd_itens": 0}
    )

    for pid, info in best_split_map.items():
        mid = int(info["mercado_id"])
        mm = next((m for m in mercados_split if int(m.id) == mid), None)
        if not mm:
            continue

        g = groups_dict[mid]
        if g["mercado"] is None:
            g["mercado"] = mm

        qtd   = float(info["qtd"])
        preco = float(info["preco"])
        linha_total = qtd * preco

        g["itens"].append({
            "pid": pid,
            "nome": nomes_produtos.get(pid, f"#{pid}"),
            "qtd": qtd,
            "preco": preco,
            "total_linha": linha_total,
            "marca": info.get("marca") or "",
            "unidade": info.get("unidade") or "",
            "tamanho": float(info.get("tamanho") or 0.0),
        })
        g["subtotal"]   += linha_total
        g["qtd_itens"]  += 1

    best_split_groups = sorted(
        groups_dict.values(),
        key=lambda G: (
            G["subtotal"],
            (G["mercado"].__dict__.get("_dist_km") or 0.0)
        )
    )

    # ------------------------------------------------------------------
    # 7) Economia entre os dois cenários (sempre positivo)
    # ------------------------------------------------------------------
    if best_single_total is not None and best_split_total is not None:
        if abs(best_single_total - best_split_total) < 1e-6:
            economia = 0.0
            economia_modo = 'igual'
        elif best_single_total < best_split_total:
            economia = best_split_total - best_single_total
            economia_modo = 'single'   # mais barato ficar em 1 mercado
        else:
            economia = best_single_total - best_split_total
            economia_modo = 'split'    # mais barato fracionar em até 3

    contexto = {
        "mercados": mercados,
        "best_single_total": best_single_total,
        "best_single_mercado": best_single_mercado,
        "best_single_map": best_single_map,
        "best_split_total": best_split_total,
        "best_split_map": best_split_map,
        "best_split_groups": best_split_groups,
        "economia": economia,
        "economia_modo": economia_modo,
        "lat": lat, "lng": lng, "raio_km": raio_km,
        "aviso": aviso,
        "nomes_produtos": nomes_produtos,
    }

    # ------------------------------------------------------------------
    # 8) Histórico + retorno
    # ------------------------------------------------------------------
    if not make_internal_call:
        try:
            hist = HistoricoComparacao(
                usuario_id=session['user_id'],
                lat=lat,
                lng=lng,
                raio_km=raio_km,
                best_single_total=best_single_total,
                best_split_total=best_split_total,
                economia=economia,
            )
            db.session.add(hist)
            db.session.commit()
        except Exception as e:
            print("Erro ao salvar histórico de comparação:", e)

        return render_template('comparar.html', **contexto)

    # chamada interna (PDF)
    return contexto
    
# ============================================================
# PDF – Mercado Único
# ============================================================


@app.route('/comparar/pdf-unico', methods=['GET'])
@login_required
def comparar_pdf_unico():
    """
    Gera um PDF com o resultado do melhor mercado único.
    Usa os dados já calculados na rota /comparar.
    """
    # Reaproveita os mesmos parâmetros de localização da tela
    lat = _to_float(request.args.get('lat'))
    lng = _to_float(request.args.get('lng'))
    raio_km = _to_float(request.args.get('raio_km')) or 10.0

    # --- Reaproveita a lógica principal chamando a função comparar() internamente? ---
    # Para não duplicar tudo, é melhor você reaproveitar o cálculo:
    # Aqui vamos chamar a mesma lógica que você usa na rota /comparar,
    # mas num cenário real o ideal seria extrair os cálculos para uma função auxiliar.
    #
    # Por enquanto, vamos supor que você já chamou /comparar e salvou
    # os resultados em sessão ou algo do tipo. Se não salvou, você pode,
    # de forma simples, refazer o cálculo aqui chamando de novo o banco.
    #
    # Para não complicar, vou assumir que você já tem:
    #   best_single_total, best_single_mercado, best_single_map, nomes_produtos
    # salvos em sessão. Se ainda não tiver, depois trabalhamos essa parte.

    best_single_total   = session.get('best_single_total')
    best_single_mercado = session.get('best_single_mercado')  # pode ser None
    best_single_map     = session.get('best_single_map', {})
    nomes_produtos      = session.get('nomes_produtos', {})

    # Se não houver dados salvos, devolve uma mensagem simples
    if not best_single_total or not best_single_map:
        return "Nenhum resultado de mercado único encontrado para gerar PDF.", 400

    # --- Cria o PDF em memória ---
    from io import BytesIO
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)

    largura, altura = A4
    y = altura - 2*cm

    # Título
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(2*cm, y, "ConfereApp - Melhor Mercado Único")
    y -= 1.5*cm

    pdf.setFont("Helvetica", 12)
    if best_single_mercado:
        pdf.drawString(2*cm, y, f"Mercado: {best_single_mercado.get('nome', 'N/D')}")
        y -= 0.7*cm
        cidade = best_single_mercado.get('cidade') or '-'
        bairro = best_single_mercado.get('bairro') or '-'
        pdf.drawString(2*cm, y, f"Cidade: {cidade} - Bairro: {bairro}")
        y -= 0.7*cm

    pdf.drawString(2*cm, y, f"Itens nesta loja: {len(best_single_map)}")
    y -= 0.7*cm
    pdf.drawString(2*cm, y, f"Total estimado: R$ {best_single_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    y -= 1*cm

    # Cabeçalho da tabela
    pdf.setFont("Helvetica-Bold", 11)
    pdf.drawString(2*cm, y, "Produto")
    pdf.drawString(8*cm, y, "Marca")
    pdf.drawString(11*cm, y, "Unid.")
    pdf.drawString(13*cm, y, "Tam.")
    pdf.drawString(15*cm, y, "Qtd")
    y -= 0.6*cm
    pdf.setFont("Helvetica", 10)

    # Linhas
    for pid, info in best_single_map.items():
        if y < 2*cm:
            pdf.showPage()
            y = altura - 2*cm
            pdf.setFont("Helvetica", 10)

        nome_prod = nomes_produtos.get(int(pid), f"#{pid}")
        marca  = info.get('marca') or ''
        unidade = info.get('unidade') or ''
        tamanho = info.get('tamanho') or 0
        qtd     = info.get('qtd') or 0

        pdf.drawString(2*cm,  y, nome_prod[:28])
        pdf.drawString(8*cm,  y, marca[:18])
        pdf.drawString(11*cm, y, str(unidade))
        pdf.drawString(13*cm, y, f"{tamanho:.2f}" if tamanho else "-")
        pdf.drawString(15*cm, y, f"{qtd:.2f}")
        y -= 0.5*cm

    pdf.showPage()
    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="confereapp_mercado_unico.pdf",
        mimetype="application/pdf"
    )


# ============================================================
# PDF – Top 3 Fracionado
# ============================================================
@app.route('/comparar/pdf-fracionado', methods=['GET'])
@login_required
def comparar_pdf_fracionado():
    """
    Gera um PDF com o resumo do Top 3 fracionado.
    Também supõe que os dados já foram calculados em /comparar
    e salvos em sessão.
    """
    from io import BytesIO
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    largura, altura = A4
    y = altura - 2*cm

    best_split_total  = session.get('best_split_total')
    best_split_groups = session.get('best_split_groups', [])
    economia          = session.get('economia')

    if not best_split_groups:
        return "Nenhum resultado fracionado encontrado para gerar PDF.", 400

    # Título
    pdf.setFont("Helvetica-Bold", 16)
    pdf.drawString(2*cm, y, "ConfereApp - Top 3 Fracionado")
    y -= 1.5*cm

    pdf.setFont("Helvetica", 12)
    pdf.drawString(2*cm, y, f"Total fracionado (até 3 mercados): R$ {best_split_total:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
    y -= 0.7*cm
    if economia is not None:
        pdf.drawString(2*cm, y, f"Economia em relação ao mercado único: R$ {economia:,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        y -= 1*cm

    pdf.setFont("Helvetica", 11)

    for grupo in best_split_groups:
        if y < 4*cm:
            pdf.showPage()
            y = altura - 2*cm
            pdf.setFont("Helvetica", 11)

        merc = grupo.get('mercado', {})
        nome   = merc.get('nome', 'Mercado')
        cidade = merc.get('cidade') or '-'
        bairro = merc.get('bairro') or '-'

        pdf.setFont("Helvetica-Bold", 11)
        pdf.drawString(2*cm, y, f"{nome} - {cidade} / {bairro}")
        y -= 0.6*cm

        pdf.setFont("Helvetica", 10)
        pdf.drawString(2*cm, y, f"Itens nesta loja: {grupo.get('qtd_itens', 0)}   Subtotal: R$ {grupo.get('subtotal', 0):,.2f}".replace(",", "X").replace(".", ",").replace("X", "."))
        y -= 0.6*cm

        # Cabeçalho simples de itens
        pdf.setFont("Helvetica-Bold", 10)
        pdf.drawString(2*cm,  y, "Produto")
        pdf.drawString(9*cm,  y, "Qtd")
        pdf.drawString(11*cm, y, "Preço")
        pdf.drawString(14*cm, y, "Total")
        y -= 0.5*cm
        pdf.setFont("Helvetica", 9)

        for it in grupo.get('itens', []):
            if y < 2*cm:
                pdf.showPage()
                y = altura - 2*cm
                pdf.setFont("Helvetica", 9)

            pdf.drawString(2*cm,  y, it.get('nome', '')[:40])
            pdf.drawString(9*cm,  y, f"{it.get('qtd', 0):.2f}")
            pdf.drawString(11*cm, y, f"R$ {it.get('preco', 0):.2f}")
            pdf.drawString(14*cm, y, f"R$ {it.get('total_linha', 0):.2f}")
            y -= 0.4*cm

        y -= 0.8*cm  # espaço entre mercados

    pdf.showPage()
    pdf.save()
    buffer.seek(0)

    return send_file(
        buffer,
        as_attachment=True,
        download_name="confereapp_top3_fracionado.pdf",
        mimetype="application/pdf"
    )



# -----------------------------------------------------------------------------
# Exportações
# -----------------------------------------------------------------------------
@app.route('/export/precos.csv')
@login_required
def export_precos_csv():
    rows = db.session.query(
        Produto.nome.label("produto"),
        Preco.marca, Preco.tamanho, Preco.unidade,
        Mercado.nome.label("mercado"),
        Preco.preco, Preco.criado_em
    ).join(Produto, Preco.produto_id == Produto.id)\
     .join(Mercado, Preco.mercado_id == Mercado.id)\
     .order_by(Produto.nome, Mercado.nome, Preco.criado_em.desc()).all()

    output = io.StringIO()
    w = csv.writer(output, lineterminator='\n')
    w.writerow(["Produto", "Marca", "Tamanho", "Unidade", "Mercado", "Preço", "Criado em"])
    for r in rows:
        w.writerow([r.produto, r.marca, r.tamanho, r.unidade, r.mercado, f"{r.preco:.2f}", r.criado_em])
    resp = Response(output.getvalue(), mimetype='text/csv; charset=utf-8')
    resp.headers['Content-Disposition'] = 'attachment; filename=precos.csv'
    return resp


# Rotas da Minha Lista de Compras
@app.route('/minha-lista')
@login_required
def minha_lista():
    uid = session['user_id']

    rows = db.session.query(
        ListaItem.id,
        Produto.nome.label('nome'),
        ListaItem.quantidade,
        ListaItem.criado_em
    ).join(Produto, ListaItem.produto_id == Produto.id) \
     .filter(ListaItem.user_id == uid) \
     .order_by(ListaItem.id.desc()) \
     .all()

    total_itens = sum(r.quantidade for r in rows)

    return render_template(
        'minha_lista.html',
        itens=rows,
        total_itens=total_itens
    )


@app.route('/minha-lista/limpar', methods=['POST'])
@login_required
def limpar_minha_lista():
    uid = session['user_id']
    ListaItem.query.filter_by(user_id=uid).delete()
    db.session.commit()
    flash('Sua lista foi limpa.', 'success')
    return redirect(url_for('minha_lista'))


@app.route('/minha-lista/remover/<int:item_id>', methods=['POST'])
@login_required
def remover_item_minha_lista(item_id):
    uid = session['user_id']
    item = ListaItem.query.filter_by(id=item_id, user_id=uid).first_or_404()
    db.session.delete(item)
    db.session.commit()
    flash('Item removido da lista.', 'success')
    return redirect(url_for('minha_lista'))

@app.route("/limpar_lista", methods=["POST"])
@login_required
def limpar_lista():
    uid = session.get("user_id")
    if not uid:
        return redirect(url_for("login"))

    # Aqui estava o erro: antes estava usuario_id
    ListaItem.query.filter_by(user_id=uid).delete()
    db.session.commit()

    flash("Lista limpa com sucesso!", "success")
    return redirect(url_for("lista"))


@app.route('/export/minha_lista.csv')
@login_required
def export_minha_lista_csv():
    uid = session['user_id']
    rows = db.session.query(
        Produto.nome,
        ListaItem.quantidade,
        ListaItem.criado_em
    ).join(Produto, ListaItem.produto_id == Produto.id) \
     .filter(ListaItem.user_id == uid) \
     .order_by(ListaItem.id.desc()) \
     .all()

    output = io.StringIO()
    w = csv.writer(output, lineterminator='\n')
    w.writerow(["Produto", "Quantidade", "Adicionado em"])

    for r in rows:
        w.writerow([
            r.nome,
            r.quantidade,
            r.criado_em.strftime('%d/%m/%Y %H:%M') if r.criado_em else ''
        ])

    resp = Response(output.getvalue(), mimetype='text/csv; charset=utf-8')
    resp.headers['Content-Disposition'] = 'attachment; filename=minha_lista.csv'
    return resp



# -----------------------------------------------------------------------------
# Admin utilidades
# -----------------------------------------------------------------------------
@app.route('/admin/counts')
def admin_counts():
    u = Usuario.query.count()
    m = Mercado.query.count()
    p = Produto.query.count()
    r = Preco.query.count()
    return f'usuarios={u} mercados={m} produtos={p} precos={r}'

def _ensure_usuario_tipo():
    try:
        cols = {c[1] for c in db.session.execute(text("PRAGMA table_info(usuarios)")).fetchall()}
        if "tipo" not in cols:
            db.session.execute(text("ALTER TABLE usuarios ADD COLUMN tipo TEXT DEFAULT 'cliente'"))
            db.session.commit()
    except Exception:
        db.session.rollback()

def _ensure_mercado_owner():
    try:
        cols = {c[1] for c in db.session.execute(text("PRAGMA table_info(mercados)")).fetchall()}
        if "owner_user_id" not in cols:
            db.session.execute(text("ALTER TABLE mercados ADD COLUMN owner_user_id INTEGER"))
            db.session.commit()
    except Exception:
        db.session.rollback()

def _ensure_produto_owner():
    try:
        cols = {c[1] for c in db.session.execute(text("PRAGMA table_info(produtos)")).fetchall()}
        if "owner_user_id" not in cols:
            db.session.execute(text("ALTER TABLE produtos ADD COLUMN owner_user_id INTEGER"))
            db.session.commit()
    except Exception:
        db.session.rollback()


def _ensure_user_columns():
    """
    Garante colunas básicas: is_admin, ativo.
    (Útil para DBs SQLite já existentes)
    """
    try:
        cols = {c[1] for c in db.session.execute(text("PRAGMA table_info(usuarios)")).fetchall()}
        changed = False
        if "is_admin" not in cols:
            db.session.execute(text("ALTER TABLE usuarios ADD COLUMN is_admin INTEGER DEFAULT 0"))
            changed = True
        if "ativo" not in cols:
            db.session.execute(text("ALTER TABLE usuarios ADD COLUMN ativo INTEGER DEFAULT 1"))
            changed = True
        if changed:
            db.session.commit()
    except Exception:
        db.session.rollback()

def _ensure_user_role_column():
    """
    Garante a coluna 'role' e promove quem tinha is_admin=1 para 'admin_geral'.
    Também garante is_admin/ativo se ainda não existirem (defensivo).
    """
    try:
        cols = {c[1] for c in db.session.execute(text("PRAGMA table_info(usuarios)")).fetchall()}
        changed = False
        if "role" not in cols:
            db.session.execute(text("ALTER TABLE usuarios ADD COLUMN role TEXT DEFAULT 'user'"))
            changed = True
        if "is_admin" not in cols:
            db.session.execute(text("ALTER TABLE usuarios ADD COLUMN is_admin INTEGER DEFAULT 0"))
            changed = True
        if "ativo" not in cols:
            db.session.execute(text("ALTER TABLE usuarios ADD COLUMN ativo INTEGER DEFAULT 1"))
            changed = True
        if changed:
            db.session.commit()
            # promove admins legados
            db.session.execute(text("UPDATE usuarios SET role='admin_geral' WHERE IFNULL(is_admin,0)=1"))
            db.session.commit()
    except Exception:
        db.session.rollback()

def _ensure_user_tipo_columns():
    """
    Garante colunas de perfil (cliente/empresa) e vínculo Mercado.
    """
    try:
        cols = {c[1] for c in db.session.execute(text("PRAGMA table_info(usuarios)")).fetchall()}
        changed = False
        if "tipo" not in cols:
            db.session.execute(text("ALTER TABLE usuarios ADD COLUMN tipo TEXT"))  # 'cliente' | 'empresa' | None (admin)
            changed = True
        if "cpf" not in cols:
            db.session.execute(text("ALTER TABLE usuarios ADD COLUMN cpf TEXT"))
            changed = True
        if "mercado_id" not in cols:
            db.session.execute(text("ALTER TABLE usuarios ADD COLUMN mercado_id INTEGER"))
            changed = True
        if changed:
            db.session.commit()
    except Exception:
        db.session.rollback()

from sqlalchemy import text

def _ensure_user_created_at():
    """
    Garante a coluna 'criado_em' em usuarios (SQLite).
    """
    try:
        cols = {c[1] for c in db.session.execute(text("PRAGMA table_info(usuarios)")).fetchall()}
        if "criado_em" not in cols:
            # cria a coluna
            db.session.execute(text("ALTER TABLE usuarios ADD COLUMN criado_em DATETIME"))
            db.session.commit()
            # preenche com timestamp atual para linhas antigas
            db.session.execute(text("UPDATE usuarios SET criado_em = CURRENT_TIMESTAMP WHERE criado_em IS NULL"))
            db.session.commit()
    except Exception:
        db.session.rollback()

from sqlalchemy import text

def _ensure_mercado_geo_columns():
    """
    Garante colunas geográficas e de auditoria em 'mercados'
    (lat, lng, dist_km, owner_user_id, criado_em) – compatível com SQLite.
    """
    try:
        cols = {c[1] for c in db.session.execute(text("PRAGMA table_info(mercados)")).fetchall()}
        alters = []
        if "lat" not in cols:
            alters.append("ALTER TABLE mercados ADD COLUMN lat REAL")
        if "lng" not in cols:
            alters.append("ALTER TABLE mercados ADD COLUMN lng REAL")
        if "dist_km" not in cols:
            alters.append("ALTER TABLE mercados ADD COLUMN dist_km REAL")
        if "owner_user_id" not in cols:
            alters.append("ALTER TABLE mercados ADD COLUMN owner_user_id INTEGER")
        if "criado_em" not in cols:
            alters.append("ALTER TABLE mercados ADD COLUMN criado_em DATETIME")

        if alters:
            with db.engine.begin() as conn:
                for sql in alters:
                    conn.execute(text(sql))
            # preenche criado_em nas linhas antigas
            db.session.execute(text("UPDATE mercados SET criado_em = CURRENT_TIMESTAMP WHERE criado_em IS NULL"))
            db.session.commit()
    except Exception:
        db.session.rollback()

from sqlalchemy import text

def _ensure_produto_columns():
    """
    Garante colunas em 'produtos' compatíveis com o modelo atual:
    - nome_canonico (TEXT)
    - categoria (TEXT)
    - owner_user_id (INTEGER)   [se você usa propriedade por empresa]
    - criado_em (DATETIME)
    """
    try:
        cols = {c[1] for c in db.session.execute(text("PRAGMA table_info(produtos)")).fetchall()}
        alters = []
        if "nome_canonico" not in cols:
            alters.append("ALTER TABLE produtos ADD COLUMN nome_canonico TEXT")
        if "categoria" not in cols:
            alters.append("ALTER TABLE produtos ADD COLUMN categoria TEXT")
        if "owner_user_id" not in cols:
            alters.append("ALTER TABLE produtos ADD COLUMN owner_user_id INTEGER")
        if "criado_em" not in cols:
            alters.append("ALTER TABLE produtos ADD COLUMN criado_em DATETIME")

        if alters:
            with db.engine.begin() as conn:
                for sql in alters:
                    conn.execute(text(sql))

        # Backfill simples do nome_canonico para linhas antigas
        # (sem depender de funções Python, funciona direto no SQLite)
        db.session.execute(text("""
            UPDATE produtos
               SET nome_canonico = LOWER(TRIM(COALESCE(nome,'')))
             WHERE nome_canonico IS NULL OR TRIM(nome_canonico) = ''
        """))
        # Preenche criado_em onde estiver nulo
        db.session.execute(text("""
            UPDATE produtos
               SET criado_em = CURRENT_TIMESTAMP
             WHERE criado_em IS NULL
        """))
        db.session.commit()
    except Exception:
        db.session.rollback()


def _ensure_listaitem_variant_columns():
    """
    Garante colunas de variante (marca, unidade, tamanho) em lista_itens (SQLite).
    """
    try:
        cols = {c[1] for c in db.session.execute(text("PRAGMA table_info(lista_itens)")).fetchall()}
        changed = False
        if "marca" not in cols:
            db.session.execute(text("ALTER TABLE lista_itens ADD COLUMN marca TEXT"))
            changed = True
        if "unidade" not in cols:
            db.session.execute(text("ALTER TABLE lista_itens ADD COLUMN unidade TEXT"))
            changed = True
        if "tamanho" not in cols:
            db.session.execute(text("ALTER TABLE lista_itens ADD COLUMN tamanho REAL"))
            changed = True
        if changed:
            db.session.commit()
    except Exception:
        db.session.rollback()


def _ensure_produto_owner_column():
    """
    Garante coluna de propriedade de produtos (empresa dona).
    """
    try:
        cols = {c[1] for c in db.session.execute(text("PRAGMA table_info(produtos)")).fetchall()}
        if "owner_user_id" not in cols:
            db.session.execute(text("ALTER TABLE produtos ADD COLUMN owner_user_id INTEGER"))
            db.session.commit()
    except Exception:
        db.session.rollback()


def seed_admin(return_str: bool = False):
    """
    Garante que exista um administrador geral.
    - Se não existir, cria: admin@confere.app / admin123
    - Se existir, garante ativo e flags corretas.
    """
    try:
        _ensure_user_columns()
        _ensure_user_role_column()

        admin = Usuario.query.filter_by(email="admin@confere.app").first()
        if admin:
            msg = "Admin já existe: admin@confere.app"
            changed = False
            if admin.ativo is False:
                admin.ativo = True; changed = True
            if not admin.is_admin:
                admin.is_admin = True; changed = True
            if admin.role != "admin_geral":
                admin.role = "admin_geral"; changed = True
            if changed:
                db.session.commit()
        else:
            admin = Usuario(
                nome="Admin",
                email="admin@confere.app",
                senha=generate_password_hash("admin123", method="pbkdf2:sha256"),
                is_admin=True,
                role="admin_geral",
                ativo=True,
            )
            db.session.add(admin)
            db.session.commit()
            msg = "Admin criado: admin@confere.app / admin123"

        if return_str:
            return msg
        print(msg)
    except (OperationalError, ProgrammingError) as e:
        msg = f"[seed admin] Estrutura ainda não pronta: {e}"
        if return_str:
            return msg
        print(msg)

from banco import HistoricoComparacao

@app.route('/historico_comparacoes')
@login_required
def historico_comparacoes():
    uid = session['user_id']

    rows = (
        HistoricoComparacao.query
        .filter(HistoricoComparacao.usuario_id == uid)   # <-- corrigido aqui
        .order_by(HistoricoComparacao.criado_em.desc())
        .limit(10)
        .all()
    )

    return render_template('historico_comparacoes.html', historicos=rows)


@app.route('/admin/seed')
def admin_seed():
    return seed_admin(return_str=True), 200

@app.route("/instalar")
def instalar():
    return render_template("instalar.html")

@app.route('/admin/users')
def admin_users():
    users = Usuario.query.order_by(Usuario.id).all()
    linhas = [f"{u.id} - {u.nome} - {u.email} - role={u.role}" for u in users]
    return "<br>".join(linhas) or "Sem usuários"

@app.route('/admin/reset_admin')
def admin_reset_admin():
    email = (request.args.get('email') or 'confereappofc@gmail.com').strip()
    senha = (request.args.get('senha') or '8n1s3v3n').strip()
    u = Usuario.query.filter_by(email=email).first()
    if not u:
        u = Usuario(nome='Admin', email=email)
        db.session.add(u)
    u.senha = generate_password_hash(senha, method="pbkdf2:sha256")
    u.is_admin = True
    u.role = "admin_geral"
    u.ativo = True
    db.session.commit()
    return f"Admin definido: {email} / senha resetada."

# -----------------------------------------------------------------------------
# API mercadinhos próximos
# -----------------------------------------------------------------------------
@app.route('/api/mercados_proximos')
@login_required
def api_mercados_proximos():
    try:
        lat = float(request.args.get('lat'))
        lng = float(request.args.get('lng'))
    except (TypeError, ValueError):
        return {"erro":"lat/lng inválidos"}, 400

    raio_km = float(request.args.get('raio_km') or (session.get('raio_km') or 10.0))
    mercados = Mercado.query.filter(Mercado.lat.isnot(None), Mercado.lng.isnot(None)).all()

    lat_deg = raio_km / 111.0
    lng_deg = raio_km / max(1e-9, (111.0 * cos(radians(lat))))
    cand = [m for m in mercados if (abs((m.lat or 0)-lat) <= lat_deg and abs((m.lng or 0)-lng) <= lng_deg)]

    resp = []
    for m in cand:
        d = haversine_km(lat, lng, m.lat, m.lng)
        if d is not None and d <= raio_km:
            resp.append({"id": m.id, "nome": m.nome, "dist_km": round(d, 2), "cidade": m.cidade, "bairro": m.bairro})

    resp.sort(key=lambda x: x["dist_km"])
    return {"mercados": resp}

# -----------------------------------------------------------------------------
# Erros
# -----------------------------------------------------------------------------
@app.errorhandler(404)
def not_found(e):
    return render_template('404.html'), 404

@app.errorhandler(500)
def server_error(e):
    return render_template('500.html'), 500



def _build_table(data):
    """Cria tabela estilizada."""
    t = Table(data, repeatRows=1)
    t.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,0), colors.darkblue),
        ('TEXTCOLOR', (0,0), (-1,0), colors.white),
        ('GRID', (0,0), (-1,-1), 0.5, colors.grey),
        ('FONTNAME', (0,0), (-1,0), 'Helvetica-Bold'),
        ('ALIGN', (0,0), (-1,-1), 'LEFT'),
        ('ROWBACKGROUNDS', (0,1), (-1,-1), [colors.whitesmoke, colors.lightgrey]),
    ]))
    return t


# 📌 EXPORTAR MERCADO ÚNICO


# -------------------------------------------------------
#  PDF MERCADO ÚNICO  (Modelo B, com logo)
# -------------------------------------------------------
@app.route("/exportar_unico_pdf")
@login_required
def exportar_unico_pdf():
    flash(
        "📄 Função disponível apenas no plano Premium.<br>"
        "Em breve você poderá gerar PDFs completos para impressão!",
        "info"
    )
    return redirect(url_for("lista"))


# -------------------------------------------------------
#  PDF TOP 3 FRACIONADO  (Modelo B, com logo)
# -------------------------------------------------------
@app.route("/exportar_fracionado_pdf")
@login_required
def exportar_fracionado_pdf():
    flash(
        "📄 Função disponível apenas no plano Premium.<br>"
        "Em breve você poderá gerar PDFs completos para impressão!",
        "info"
    )
    return redirect(url_for("lista"))

# -----------------------------------------------------------------------------
# Bootstrap
# -----------------------------------------------------------------------------
if __name__ == '__main__':
    with app.app_context():
        db.create_all()
        _ensure_user_columns()
        _ensure_user_role_column()
        _ensure_user_tipo_columns()
        _ensure_produto_owner_column()
        _ensure_usuario_tipo()
        _ensure_mercado_owner()
        _ensure_user_created_at()
        _ensure_mercado_geo_columns()
        _ensure_produto_columns()
        ensure_lista_itens_columns()
        seed_admin(return_str=False)

    app.run(host="0.0.0.0", port=5000)


