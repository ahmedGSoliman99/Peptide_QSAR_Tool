# Deploy the Peptide QSAR Tool as a Web App

This project can run online as a Streamlit web app. GitHub Pages cannot run the Python application itself because GitHub Pages only serves static HTML/CSS/JavaScript files. Use Streamlit Community Cloud for the live app.

## Recommended Live-App Deployment

1. Open Streamlit Community Cloud:

   https://share.streamlit.io/

2. Sign in with GitHub using the account:

   `ahmedGSoliman99`

3. Choose **New app**.

4. Select this repository:

   `ahmedGSoliman99/Peptide_QSAR_Tool`

5. Use these deployment settings:

   - Branch: `main`
   - Main file path: `app.py`
   - Python version: read from `runtime.txt` (`python-3.11`)
   - Dependencies: read from `requirements.txt`

6. Optional custom subdomain:

   `peptide-qsar-tool`

   If available, the app URL will be:

   https://peptide-qsar-tool.streamlit.app/

7. Click **Deploy**.

## Notes

- The app will run in the browser and will not require users to download the Windows executable.
- Public users can upload peptide datasets and use the app from the browser.
- Saved model files on the cloud instance should be treated as temporary; for permanent sharing, export model files and keep local copies.
- Large training jobs may be slower on free community hosting than on a local computer.

