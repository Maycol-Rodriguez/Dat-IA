// Curso Dat-IA — vanilla JS, sin dependencias. Toggle de tema, sidebar activa, quiz, tabs.

(function themeToggle() {
  const KEY = "dat-ia-curso-theme"; // "light" | "dark"
  const root = document.documentElement;
  const saved = localStorage.getItem(KEY);
  if (saved) root.setAttribute("data-theme", saved);

  function currentIsDark() {
    const attr = root.getAttribute("data-theme");
    if (attr) return attr === "dark";
    return window.matchMedia("(prefers-color-scheme: dark)").matches;
  }

  function syncLabel(btn) {
    if (!btn) return;
    btn.textContent = currentIsDark() ? "☀️ Modo claro" : "🌙 Modo oscuro";
  }

  document.addEventListener("DOMContentLoaded", () => {
    const btn = document.getElementById("theme-toggle");
    syncLabel(btn);
    if (btn) {
      btn.addEventListener("click", () => {
        const next = currentIsDark() ? "light" : "dark";
        root.setAttribute("data-theme", next);
        localStorage.setItem(KEY, next);
        syncLabel(btn);
      });
    }
  });
})();

// Marca como activo el link de la sidebar cuya href coincide con la página actual.
document.addEventListener("DOMContentLoaded", () => {
  const current = location.pathname.split("/").pop() || "index.html";
  document.querySelectorAll(".sidebar nav a").forEach((a) => {
    const href = a.getAttribute("href");
    if (href === current) a.classList.add("active");
  });
});

// Acordeón de quiz: cada .quiz-item alterna .open al hacer clic en .quiz-q.
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".quiz-item").forEach((item) => {
    const q = item.querySelector(".quiz-q");
    if (!q) return;
    q.addEventListener("click", () => {
      item.classList.toggle("open");
    });
  });
});

// Tabs: dentro de cada .tabs, los .tab-btn activan el .tab-panel con el mismo data-tab.
document.addEventListener("DOMContentLoaded", () => {
  document.querySelectorAll(".tabs").forEach((tabs) => {
    const buttons = tabs.querySelectorAll(".tab-btn");
    const panels = tabs.querySelectorAll(".tab-panel");
    buttons.forEach((btn) => {
      btn.addEventListener("click", () => {
        const target = btn.getAttribute("data-tab");
        buttons.forEach((b) => b.classList.toggle("active", b === btn));
        panels.forEach((p) =>
          p.classList.toggle("active", p.getAttribute("data-tab") === target)
        );
      });
    });
  });
});
