// Bundles the UI to web/static/app.js, which autopsy.api serves at /.
// Same origin as the API, so no CORS and no second dev server.
//
//   npm install && npm run build      then: uvicorn autopsy.api:app
//   npm run watch                     rebuild on save
import { build, context } from "esbuild";
import { createHash } from "node:crypto";
import { readFileSync, writeFileSync } from "node:fs";
import { fileURLToPath } from "node:url";
import { dirname, resolve } from "node:path";

const root = resolve(dirname(fileURLToPath(import.meta.url)), "..");
const options = {
  entryPoints: [resolve(root, "web/src/entry.jsx")],
  outfile: resolve(root, "web/static/app.js"),
  bundle: true,
  minify: true,
  // IIFE, not ESM: browsers refuse module scripts from file://, and the
  // static showcase has to survive being double-clicked.
  format: "iife",
  target: ["es2020"],
  jsx: "automatic",
  loader: { ".jsx": "jsx" },
  define: { "process.env.NODE_ENV": '"production"' },
  logLevel: "info",
};

// The page asks for app.js?v=<hash of app.js>, rewritten on every build.
//
// Without it a browser keeps serving the bundle it already has: the API
// moves on, the cached page does not, and the two disagree in ways that read
// as server bugs. That is not hypothetical — it is how a working job store
// came to look like one that loses jobs, because a cached bundle polled an
// endpoint whose contract had changed under it. A URL that changes with the
// content ends the whole class: the browser cannot serve a stale bundle for
// a filename it has never seen.
function stampIndex() {
  const bundle = resolve(root, "web/static/app.js");
  const page = resolve(root, "web/static/index.html");
  const hash = createHash("sha256").update(readFileSync(bundle)).digest("hex").slice(0, 12);
  const html = readFileSync(page, "utf8")
    .replace(/src="\.\/app\.js(\?v=[0-9a-f]+)?"/, `src="./app.js?v=${hash}"`);
  writeFileSync(page, html);
  console.log(`  index.html -> app.js?v=${hash}`);
}

if (process.argv.includes("--watch")) {
  const ctx = await context({
    ...options,
    plugins: [{ name: "stamp", setup: (b) => b.onEnd(stampIndex) }],
  });
  await ctx.watch();
  console.log("watching web/src, model_autopsy.jsx, ui_connector.jsx…");
} else {
  await build(options);
  stampIndex();
}
