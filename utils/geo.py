# utils/geo.py
# -*- coding: utf-8 -*-
import requests
from banco import db, GeoCache

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def geocode_with_cache(chave_endereco: str):
    """
    Faz geocodificação usando Nominatim + cache no banco.

    A ideia é receber um endereço o mais completo possível, incluindo:
        logradouro, número, bairro, cidade, UF e, se tiver, CEP.

    Exemplo de chave_endereco:
        "Av João Manoel da Silva 365, Centro, Toritama, PE, 55125-000"

    Retorna:
        (lat, lng) -> floats
        ou (None, None) se não encontrar.
    """
    chave = (chave_endereco or "").strip()
    if not chave:
        return None, None

    # 1) Procura no cache
    cached = GeoCache.query.filter_by(chave=chave).first()
    if cached:
        return cached.lat, cached.lng

    # 2) Consulta Nominatim (OpenStreetMap)
    params = {
        "q": chave,
        "format": "json",
        "limit": 1,
        "addressdetails": 0,
        "countrycodes": "br",  # força resultados somente no Brasil
    }

    headers = {
        "User-Agent": "ConfereApp/1.0 (contato@confereapp.com)"
    }

    try:
        resp = requests.get(
            NOMINATIM_URL,
            params=params,
            headers=headers,
            timeout=10
        )
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        print("[GEO] Erro consultando Nominatim:", e)
        return None, None

    if not data:
        print("[GEO] Nominatim não retornou resultados para:", chave)
        return None, None

    try:
        lat = float(data[0]["lat"])
        lng = float(data[0]["lon"])
    except (KeyError, ValueError) as e:
        print("[GEO] Resposta inesperada do Nominatim:", e, data[0])
        return None, None

    # 3) Salva no cache (se falhar, não deixa o app cair)
    try:
        novo = GeoCache(chave=chave, lat=lat, lng=lng)
        db.session.add(novo)
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print("[GEO] Erro salvando no cache:", e)

    return lat, lng
