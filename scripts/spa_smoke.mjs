import { chromium } from "playwright";

const url = process.env.SPA_URL || "http://127.0.0.1:8768/";
const browser = await chromium.launch();
const page = await browser.newPage({ viewport: { width: 1280, height: 800 } });
const errors = [];
page.on("console", (msg) => {
  if (msg.type() === "error") errors.push(msg.text());
});
page.on("pageerror", (err) => errors.push(String(err)));

await page.goto(url, { waitUntil: "networkidle", timeout: 15000 });
await page.waitForTimeout(800);

const newChatBtn = await page.locator("text=+ New chat").count();
const placeholder = await page.locator("text=Start a conversation").count();
const composer = await page.locator('textarea').count();
const title = await page.title();

await page.screenshot({
  path: "gui-test-screenshots/spa_smoke.png",
  fullPage: false,
});

const result = {
  title,
  newChatBtnVisible: newChatBtn > 0,
  emptyStateVisible: placeholder > 0,
  composerVisible: composer > 0,
  consoleErrors: errors,
};
console.log(JSON.stringify(result, null, 2));

await browser.close();
// Non-zero exit if anything looks wrong.
if (!(newChatBtn > 0 && composer > 0) || errors.length) {
  process.exit(1);
}
