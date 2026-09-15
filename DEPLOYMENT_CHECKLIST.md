# V2 Deployment Checklist

- [ ] Revoke/regenerate the Gemini API key that was previously stored in a local `.env` file.
- [ ] Confirm no `.env` or secrets file is tracked by Git.
- [ ] Confirm `streamlit run app/streamlit_app.py` works locally.
- [ ] Confirm demo mode works with no external credentials.
- [ ] Confirm uploaded CSV workflow works.
- [ ] Add `GEMINI_API_KEY` to Streamlit Cloud Secrets only if AI/RAG is enabled.
- [ ] Do not commit large model/data artifacts without checking repository size.
- [ ] Push to a new V2 GitHub repository.
- [ ] Deploy `app/streamlit_app.py` on Streamlit Cloud.
- [ ] Test the public URL in a clean browser session.
