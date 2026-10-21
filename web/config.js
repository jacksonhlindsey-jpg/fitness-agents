// Where the agents' API lives.
// Locally the page and the API are served by the same server (uv run fitness-web), so this is "".
// On GitHub Pages the API is the server hosted on Render.
window.FITNESS_API_BASE = location.hostname.endsWith("github.io")
  ? "https://fitness-agents.onrender.com"
  : "";
