# fix_admin.py
from app import app, db, Usuario

with app.app_context():
    u = Usuario.query.filter_by(email="admin@confere.app").first()
    if not u:
        print("Admin não encontrado.")
    else:
        u.is_admin = True
        u.ativo = True
        db.session.commit()
        print("Admin ajustado: is_admin=True, ativo=True")
