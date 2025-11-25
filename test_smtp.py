# test_smtp.py (versão com diagnóstico)
import os, smtplib
from email.mime.text import MIMEText
from pathlib import Path

print("[DEBUG] CWD:", os.getcwd())
print("[DEBUG] Arquivos na pasta:", os.listdir())

# Carrega .env explicitamente do mesmo diretório deste arquivo
dotenv_path = Path(__file__).with_name('.env')
print("[DEBUG] dotenv_path existe?", dotenv_path.exists(), "->", dotenv_path)

try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=dotenv_path)
    print("[DEBUG] python-dotenv carregado.")
except Exception as e:
    print("[DEBUG] python-dotenv NÃO instalado ou falhou:", e)

# Leia variáveis
SMTP_HOST = os.getenv("SMTP_HOST")
SMTP_PORT = int(os.getenv("SMTP_PORT") or 0)
SMTP_USER = os.getenv("SMTP_USER")
SMTP_PASS = os.getenv("SMTP_PASS")
SMTP_FROM = os.getenv("SMTP_FROM") or SMTP_USER
DESTINO   = os.getenv("SMTP_TEST_TO") or SMTP_USER

print("[DEBUG] SMTP_HOST:", SMTP_HOST)
print("[DEBUG] SMTP_PORT:", SMTP_PORT)
print("[DEBUG] SMTP_USER:", SMTP_USER)
print("[DEBUG] SMTP_FROM:", SMTP_FROM)
print("[DEBUG] DESTINO:", DESTINO)

assert SMTP_HOST and SMTP_PORT and SMTP_USER and SMTP_PASS and SMTP_FROM, (
    "Variáveis SMTP_* ausentes. Verifique seu .env (SMTP_HOST, SMTP_PORT, SMTP_USER, SMTP_PASS, SMTP_FROM)."
)

# Envio
msg = MIMEText("Teste de envio SMTP do ConfereApp ✅", "plain", "utf-8")
msg["Subject"] = "Teste SMTP - ConfereApp"
msg["From"] = SMTP_FROM
msg["To"] = DESTINO

print(f"[SMTP] Conectando em {SMTP_HOST}:{SMTP_PORT} como {SMTP_USER}")
with smtplib.SMTP(SMTP_HOST, SMTP_PORT) as s:
    s.set_debuglevel(1)
    s.ehlo()
    if SMTP_PORT in (587, 25):
        s.starttls()
        s.ehlo()
    s.login(SMTP_USER, SMTP_PASS)
    s.sendmail(SMTP_FROM, [DESTINO], msg.as_string())
print("[SMTP] Enviado com sucesso para:", DESTINO)
