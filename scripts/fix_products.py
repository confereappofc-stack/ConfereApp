# scripts/fix_products.py
# Corrige nomes duplicados de Produto unificando por nome_canonico
import os
import sys
from pathlib import Path

# --- Garantir que o Python enxergue a raiz do projeto (onde ficam app.py e banco.py)
ROOT = Path(__file__).resolve().parent.parent  # ...\confereapp
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Agora podemos importar normalmente
try:
    from app import app
    from banco import db, Produto
    from sqlalchemy import func
except Exception as e:
    print("Erro importando app/banco:", e)
    sys.exit(1)


def run(apply=False):
    """
    Se apply=False: apenas mostra o que seria feito (dry-run)
    Se apply=True: aplica as correções no banco
    """
    with app.app_context():
        print(">> Produtos antes:", db.session.query(Produto).count())

        # Pega grupos com mesmo nome_canonico
        grupos = (
            db.session.query(Produto.nome_canonico,
                             func.count(Produto.id).label('qt'))
            .group_by(Produto.nome_canonico)
            .having(func.count(Produto.id) > 1)
            .all()
        )

        if not grupos:
            print("Nenhum nome_canonico duplicado encontrado. Nada a fazer.")
            return

        print(f"Encontrados {len(grupos)} nomes canônicos duplicados.")
        total_removidos = 0

        for canon, qt in grupos:
            # Ordena por id para manter o menor id como "oficial"
            produtos = (
                db.session.query(Produto)
                .filter(Produto.nome_canonico == canon)
                .order_by(Produto.id.asc())
                .all()
            )
            kept = produtos[0]
            dups = produtos[1:]
            if not dups:
                continue

            print(f"\n- Canon '{canon}': manter #{kept.id} '{kept.nome}', remover {len(dups)} duplicado(s)")
            for d in dups:
                print(f"  * apagar #{d.id} '{d.nome}'")

            if apply:
                # Remover duplicados
                for d in dups:
                    db.session.delete(d)
                total_removidos += len(dups)

        if apply:
            db.session.commit()
            print(f"\n>> Removidos {total_removidos} duplicados.")
            print(">> Produtos depois:", db.session.query(Produto).count())
        else:
            print("\n(DRY-RUN) Nada foi alterado. Use --apply para aplicar as mudanças.")


if __name__ == "__main__":
    # Windows: `python scripts\fix_products.py --apply` para aplicar
    apply = ("--apply" in sys.argv)
    run(apply=apply)
