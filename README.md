# Dashboards Tickets Grupo Continental

Dashboard de análisis de tickets Jira con métricas operativas, NLP y exportación de reportes interactivos.

## Ejecutar en local

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Desplegar en Streamlit Cloud

1. Sube este repositorio a tu GitHub.
2. Ve a [share.streamlit.io](https://share.streamlit.io).
3. Inicia sesión con GitHub.
4. Click en **"New app"** → selecciona el repo, branch `main`, archivo `app.py`.
5. Elige "Advanced settings" y deja `requirements.txt` como dependencias.
6. Deploy.

**Nota:** En la nube necesitarás subir un archivo Jira (CSV/Excel) desde el sidebar para ver datos. Los archivos en `data/` no se suben por defecto (son locales).
