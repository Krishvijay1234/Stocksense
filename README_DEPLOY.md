# Permanent local run and deployment

## Run locally forever

### Backend
Run PowerShell:

```powershell
powershell -ExecutionPolicy Bypass -File .\run_server.ps1
```

### Frontend
Run in a separate terminal:

```powershell
cd frontend
npm run dev -- --host 0.0.0.0
```

## Deploy to Render

1. Push this repo to GitHub.
2. Create a new Web Service on Render.
3. Set build command:

```bash
pip install -r requirements.txt
cd frontend && npm install && npm run build
```

4. Set start command:

```bash
gunicorn --bind 0.0.0.0:$PORT backend_app:app
```

5. Add environment variable:
   - VITE_API_URL = /

## Deploy to Railway

1. Push to GitHub.
2. Create a Railway project and deploy.
3. Use the existing Procfile.
