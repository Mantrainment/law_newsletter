# Diritto del Lavoro Newsletter Agent

Newsletter settimanale via email con articoli da riviste di diritto del lavoro italiane, europee e internazionali.

## Categorie
- **Diritto del Lavoro** — Dottrina e legislazione italiana
- **Relazioni Industriali** — Diritto sindacale, contrattazione collettiva
- **Diritto Europeo e Comparato del Lavoro** — UE, comparato, internazionale
- **Sicurezza sul Lavoro e Previdenza** — Sicurezza, salute, previdenza sociale
- **Giurisprudenza del Lavoro** — Sentenze, massime, commenti

## Fonti
29 riviste giuridiche tra cui: Rivista Giuridica del Lavoro, Rivista Italiana di Diritto del Lavoro, Lavoro e Diritto, European Labour Law Journal, Industrial Law Journal, Droit Social, e molte altre.

Le fonti vengono interrogate tramite RSS, scraping HTML, e Google Scholar come fallback.

## Funzionalità
- **[★ Star]** su un articolo per riceverne di simili nelle prossime newsletter
- **[Reject]** su un articolo per filtrare articoli simili in futuro
- **Serendipity Discovery** — esplora automaticamente campi giuridici adiacenti ai tuoi interessi
- **Deduplicazione** — gli articoli già inviati non vengono riproposti
- **Filtraggio semantico** — usa embeddings per confrontare articoli nuovi con quelli rifiutati/preferiti

---

## Opzione 1: GitHub Actions (Consigliato — Gratuito & Automatico)

Nessun server necessario. GitHub lo esegue automaticamente ogni settimana.

### Setup:

1. **Crea un repo GitHub** e carica questi file:
   ```
   your-repo/
   ├── law_newsletter_agent.py
   ├── feedback.json
   ├── requirements.txt
   └── .github/workflows/newsletter.yml
   ```

2. **Aggiungi i secrets** nel repo:
   - Vai a: Settings → Secrets → Actions → New repository secret
   - Aggiungi questi secrets:
     | Nome | Valore |
     |------|--------|
     | `EMAIL_SENDER` | tua_email@gmail.com |
     | `EMAIL_PASSWORD` | App Password Gmail |
     | `EMAIL_RECIPIENT` | tua_email@gmail.com |
     | `SERPAPI_KEY` | (opzionale) chiave API SerpAPI per Google Scholar |

3. **Testa**: 
   - Vai a: Actions → "Weekly Law Newsletter" → Run workflow

La newsletter verrà inviata automaticamente ogni lunedì alle 8:00 AM UTC.

---

## Opzione 2: Esecuzione Locale

### Installa & Configura
```bash
pip install -r requirements.txt
# Modifica EMAIL_CONFIG nello script
```

### Esegui una volta (test)
```bash
python law_newsletter_agent.py --run-once
```

### Esegui continuativamente
```bash
pip install schedule
python law_newsletter_agent.py
```

---

## Gmail App Password
1. Vai a [Google Account Security](https://myaccount.google.com/security)
2. Abilita la verifica in 2 passaggi
3. Vai a App Password → Genera nuova
4. Usa la password di 16 caratteri generata
