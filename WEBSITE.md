# SMS AC website

Run from D:\SEproject:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-web.txt
.\Start-Website.ps1
```

Open http://127.0.0.1:8000. Create an account; passwords use Django's password hashing. The website keeps the discussion in a server-side database session and restricts report downloads to that session. Login, sign-up, CSRF protection and Google OAuth redirects are tested. Actual Google account consent requires credentials and has not been tested.

## Connect the existing trained LLM1

The regenerated `kaggle/llm1_chat_app.ipynb` adds explicit `/website_chat` and `/website_report` Gradio APIs and embeds the IEEE 830 outline from `src/sms_ac/srs.py` during bootstrap. Existing trained weights are unchanged. A previous Gradio app version does not expose these new endpoints.

Local GPU inference is now the default; see `LOCAL_INFERENCE.md`. For the optional remote mode, set `SMS_INFERENCE=remote` and `LLM1_URL` to the new running Kaggle URL before starting Django directly with `website/manage.py runserver`. The local launcher deliberately forces local mode. A previous Gradio runtime cannot provide the new SRS APIs.

```powershell
$env:LLM1_URL = 'https://YOUR-CURRENT-RUNTIME.gradio.live'
.\Start-Website.ps1
```

## Enable Google sign-in

Create a Google Cloud OAuth client of type Web application. Set its authorised redirect URI to `http://127.0.0.1:8000/accounts/google/login/callback/` for local testing (use your exact HTTPS domain for deployment). Configure the consent screen and test users as applicable. Set `GOOGLE_CLIENT_ID` and `GOOGLE_CLIENT_SECRET` in the server environment and restart. Do not commit secrets. See https://docs.allauth.org/en/latest/socialaccount/providers/google.html.

## Public deployment

This is a local development website, not a deployed public service. Before deployment, use a production WSGI server, configure HTTPS, `SMS_DEBUG=0`, a strong `DJANGO_SECRET_KEY`, `SMS_ALLOWED_HOSTS` and a real email backend with mandatory verified emails and password recovery. Protect the model endpoint with backend credentials or private networking; a public Gradio URL can otherwise bypass the website login. The current runtime must not be treated as a private production inference endpoint. Move model calls to background jobs before multi-worker use; current report generation is synchronous. Expired sessions are cleaned with Django's `clearsessions` command; downloaded reports also need a retention policy.

The output is an IEEE 830-1998 structured **draft**, not an assertion of certification. The outline retains feasibility, timeline and traceability in appendices; LLM2 remains responsible for SDLC selection.

## Checks

```powershell
.\.venv\Scripts\python.exe website/manage.py test workspace
.\.venv\Scripts\python.exe website/manage.py check
```
