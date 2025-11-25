# corrigir_produtos_duplicados.py
# Junta produtos duplicados por nome (normalizado) e realoca PREÇOS para o produto representante.
# Não muda valores, só troca o produto_id nos preços e remove o duplicado.

from app import app
from banco import db, Produto, Preco

def canon(s: str) -> str:
    # normaliza: tira espaços extras e deixa minúsculas
    return " ".join((s or "").strip().lower().split())

with app.app_context():
    produtos = Produto.query.all()
    por_canon = {}
    mesclados = 0

    for p in produtos:
        c = canon(p.nome)
        if c in por_canon:
            manter = por_canon[c]  # representante

            # Se por alguma razão caiu no mesmo registro, ignora
            if manter.id == p.id:
                continue

            print(f"🔁 Mesclando '{p.nome}' (id {p.id}) -> '{manter.nome}' (id {manter.id})")

            # Move todos os preços do duplicado para o representante
            Preco.query.filter(Preco.produto_id == p.id).update(
                {"produto_id": manter.id},
                synchronize_session=False
            )

            # Remove o produto duplicado
            db.session.delete(p)
            mesclados += 1
        else:
            por_canon[c] = p

    db.session.commit()
    print("✅ Concluído.")
    print("Produtos:", Produto.query.count(), "| Preços:", Preco.query.count(), "| Duplicados mesclados:", mesclados)
