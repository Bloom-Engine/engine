"use strict";

const fs = require("node:fs");
const path = require("node:path");
const http = require("node:http");

async function serve(directory, port) {
  const root = await fs.promises.realpath(directory);
  const types = { ".html": "text/html; charset=utf-8", ".js": "text/javascript", ".mjs": "text/javascript", ".wasm": "application/wasm", ".json": "application/json", ".png": "image/png", ".txt": "text/plain; charset=utf-8" };
  const server = http.createServer(async (request, response) => {
    try {
      if (!["GET", "HEAD"].includes(request.method)) { response.writeHead(405).end(); return; }
      const route = decodeURIComponent(new URL(request.url, "http://localhost").pathname);
      const file = await fs.promises.realpath(path.resolve(root, "." + (route === "/" ? "/index.html" : route)));
      const relative = path.relative(root, file);
      if (relative.startsWith(".." + path.sep) || relative === ".." || path.isAbsolute(relative)) { response.writeHead(403).end(); return; }
      const stat = await fs.promises.stat(file);
      if (!stat.isFile()) { response.writeHead(404).end(); return; }
      response.writeHead(200, { "Content-Type": types[path.extname(file)] || "application/octet-stream", "Content-Length": stat.size, "Cache-Control": "no-store" });
      if (request.method === "HEAD") response.end();
      else fs.createReadStream(file).on("error", () => response.destroy()).pipe(response);
    } catch { response.writeHead(404).end(); }
  });
  await new Promise((resolve, reject) => {
    server.once("error", reject);
    server.listen(port, "127.0.0.1", resolve);
  });
  console.log(`Bloom: http://127.0.0.1:${server.address().port} (Ctrl+C to stop)`);
  return server;
}

module.exports = { serve };
