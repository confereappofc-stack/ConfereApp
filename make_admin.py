from app import app, db, Usuario

email_alvo = "usuario@confere.app"   # <-- coloque aqui o e-mail do usuário

with app.app_context():
    u = Usuario.query.filter_by(email=email_alvo).first()
    if u:
        u.is_admin = True
        db.session.commit()
        print(f"{u.email} agora é administrador.")
    else:
        print("Usuário não encontrado.")
