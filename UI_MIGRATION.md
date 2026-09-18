# UI Migration Notes

**Author:** Claire Yuan  
**Advisor:** Dr. Qingyang Xiao

This repository incorporates the visual language of the attached TanStack/React UI design into the Streamlit application.

## What was migrated

- Warm editorial paper background (`#faf8f3`)
- Dark ink text and fine rule/border system
- Orange accent treatment
- JetBrains Mono body/interface typography
- Instrument Serif display/report typography
- Minimal square brand mark and version label
- Horizontal ticker treatment
- Large editorial hero typography
- Uppercase micro-labels and section kickers
- Rule-based tab and card treatments
- Editorial result/report layout
- Principles and footer structure
- Mobile-responsive layout adjustments

## Why the React app itself is not run inside Streamlit

The supplied design is a separate TanStack/React application with Node/server/Supabase-oriented dependencies and domain-specific legislation/news services. Streamlit Community Cloud is launching this project from `app.py`. Running both stacks would make deployment heavier and would reintroduce unnecessary services unrelated to Claire Yuan's court-notice analysis prototype.

The migration therefore translates the visual system into native Streamlit + CSS while preserving the existing Python inference pipeline as the functional source of truth.

## What was intentionally not copied

- AP News/legislation feeds
- Congress/OpenStates lookup services
- Supabase database migrations
- Watchlist account/server functions
- MCP/Cloudflare/TanStack server configuration
- Any `.env` content from the supplied UI archive

These elements are unrelated to the AI Court-Bill Impact Analyzer and are not required for Streamlit Community Cloud deployment.
