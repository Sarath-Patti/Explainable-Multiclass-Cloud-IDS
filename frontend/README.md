# Explainable Multiclass Cloud IDS - Frontend

Production-ready React + TypeScript + Vite + Tailwind CSS frontend interface for Explainable Multiclass Cloud Intrusion Detection System.

## Architecture

```
frontend/
├── src/
│   ├── assets/        # Static media assets
│   ├── components/    # Reusable UI components (Navbar, Footer, HealthStatus)
│   ├── pages/         # Page views (Home, Dashboard, Upload)
│   ├── services/      # Axios HTTP client & API integrations
│   ├── hooks/         # Custom React hooks (useHealth)
│   ├── types/         # TypeScript interface definitions
│   ├── App.tsx        # Main application layout & React Router config
│   ├── main.tsx       # React DOM entrypoint
│   └── index.css      # Tailwind directives & glassmorphism theme
├── public/            # Public static assets
├── package.json
├── tsconfig.json
├── vite.config.ts
├── tailwind.config.js
├── postcss.config.js
└── README.md
```

## Setup & Execution Instructions

1. **Navigate to Frontend Directory**:
   ```bash
   cd frontend
   ```

2. **Install Dependencies**:
   ```bash
   npm install
   ```

3. **Start Development Server**:
   ```bash
   npm run dev
   ```

   The application will be accessible at `http://localhost:5173`.

4. **Build Production Bundle**:
   ```bash
   npm run build
   ```
