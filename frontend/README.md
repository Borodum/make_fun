
  # Joke Generator Website

  This is a small frontend for the "Joke Generator" web app — a UI for uploading images and generating jokes/captions.

  The original design is available on Figma: https://www.figma.com/design/hV5FpAsjdx3nApFfg3QQSC/Joke-Generator-Website

  ## Quick start

  Requirements:
  - Node.js and npm (or yarn)

  Install dependencies:

  ```bash
  npm i
  ```

  Run in development mode:

  ```bash
  npm run dev
  ```

  Common additional commands (if present in `package.json`):

  ```bash
  npm run build    # build the app for production
  npm run preview  # preview the built app locally
  ```

  ## Project structure (important files and folders)

  - `index.html` — Vite entry HTML.
  - `package.json` — dependencies and npm scripts.
  - `vite.config.ts` — Vite configuration.
  - `src/` — application source code:
    - `main.tsx` / `App.tsx` — React entry and root component.
    - `index.css`, `styles/globals.css` — global styles.
    - `components/` — UI components:
      - `ImageUpload.tsx` — image upload component.
      - `JokeCard.tsx` — card that shows image and joke/caption.
      - `StarRating.tsx` — rating component.
      - `figma/ImageWithFallback.tsx` — helper image component.
      - `ui/` — collection of small reusable UI utilities and primitives (buttons, inputs, dialogs, etc.).
    - `guidelines/Guidelines.md` — project notes and design/style guidelines.

  - `frontend/README.md` — this file.

  ## Architecture overview

  The project is organized modularly: small reusable UI primitives live in `src/components/ui/`, while larger business components live in `src/components/`. This simplifies testing and reuse.

  Styles are applied using global CSS and (where used) CSS modules in `src/styles/` and `index.css`.

  