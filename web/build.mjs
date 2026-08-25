// Bundles the UI to web/static/app.js, which autopsy.api serves at /.
// Same origin as the API, so no CORS and no second dev server.
//
//   npm install && npm run build      then: uvicorn autopsy.api:app
//   npm run watch                     rebuild on save
import { build, context } from "esbuild";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const options = {
  entryPoints: [resolve(root, "web/src/entry.jsx")],
  outfile: resolve(root, "web/static/app.js"),
  bundle: true,
  minify: true,
  format: "esm",
  target: ["es2020"],
  jsx: "automatic",
  loader: { ".jsx": "jsx" },
  define: { "process.env.NODE_ENV": '"production"' },
  logLevel: "info",
};

if (process.argv.includes("--watch")) {
  const ctx = await context(options);
  await ctx.watch();
  console.log("watching web/src, model_autopsy.jsx, ui_connector.jsx…");
} else {
  await build(options);
}
