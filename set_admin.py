# set_admin.py
from app import app, db, seed_admin, Usuario  # agora importa Usuario também

with app.app_context():
    print(seed_admin(return_str=True))
    u = Usuario.query.filter_by(email="admin@confere.app").first()
    if u:
        print("Usuário admin encontrado:")
        print(" - Nome:", u.nome)
        print(" - E-mail:", u.email)
        print(" - is_admin:", u.is_admin)
        print(" - ativo:", u.ativo)
