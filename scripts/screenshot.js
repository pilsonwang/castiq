// Usage: node scripts/screenshot.js docs/index.html docs/assets/dashboard.png
// Captures the top of the dashboard (header, KPIs, market section) at 2x for the README.
const path = require("path");
const { chromium } = require("playwright");

(async () => {
  const [input = "docs/index.html", output = "docs/assets/dashboard.png"] = process.argv.slice(2);
  const browser = await chromium.launch();
  const page = await browser.newPage({ viewport: { width: 1280, height: 1500 }, deviceScaleFactor: 2,
                                       colorScheme: "light" });
  await page.goto("file://" + path.resolve(input));
  await page.waitForTimeout(2500);
  await page.screenshot({ path: output, clip: { x: 0, y: 0, width: 1280, height: 1500 } });
  await browser.close();
  console.log("wrote", output);
})();
