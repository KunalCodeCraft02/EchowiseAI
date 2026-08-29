const GENERIC_ERROR = "Something went wrong. Please try again.";

// The LAST line of defence against a developer-facing error reaching a clinician.
//
// Routes are expected to send a written sentence in `detail`, but three things arrive here that
// are not one, and every one of them used to be printed on screen verbatim:
//   - FastAPI validation failures, where `detail` is an ARRAY of {loc, msg, type} objects and
//     rendered as the string "[object Object]";
//   - an upstream provider's error body forwarded as-is, which is raw JSON;
//   - proxy/gateway HTML for a 502 or 504, when the response is not JSON at all.
// Anything that is not a plain sentence is replaced with GENERIC_ERROR. Losing a detail a
// clinician could not have acted on costs nothing; showing them a JSON blob suggests their
// report is corrupt when the actual fault is a server-side one they cannot see.
async function friendlyError(res) {
  let detail;
  try { detail = (await res.json()).detail; } catch { return GENERIC_ERROR; }

  if (typeof detail !== "string") return GENERIC_ERROR;   // array, object, null
  const text = detail.trim();
  if (!text) return GENERIC_ERROR;
  if (text.startsWith("{") || text.startsWith("[")) return GENERIC_ERROR;   // serialized payload
  if (/"[a-z_]+"\s*:/i.test(text)) return GENERIC_ERROR;                    // JSON embedded midway
  if (/\bTraceback\b|\bat [A-Za-z$_]+ \(/.test(text)) return GENERIC_ERROR; // stack trace
  return text;
}

// Shared helpers used across all pages.
// Points at the Railway/Render-hosted FastAPI backend -- this frontend is deployed
// separately (Vercel) from the API, unlike local dev where FastAPI serves both.
const API = {
  base: "https://echowiseai-backend.onrender.com",
  token() { return localStorage.getItem("access_token"); },
  doctor() { try { return JSON.parse(localStorage.getItem("doctor")); } catch { return null; } },

  setSession(token, doctor) {
    localStorage.setItem("access_token", token);
    localStorage.setItem("doctor", JSON.stringify(doctor));
  },
  clearSession() {
    localStorage.removeItem("access_token");
    localStorage.removeItem("doctor");
  },
  requireAuth() {
    if (!this.token()) window.location.href = "/login";
  },

  async request(path, options = {}) {
    const headers = options.headers || {};
    if (this.token()) headers["Authorization"] = `Bearer ${this.token()}`;
    if (options.json) {
      headers["Content-Type"] = "application/json";
      options.body = JSON.stringify(options.json);
    }
    const res = await fetch(this.base + path, { ...options, headers });
    if (res.status === 401) {
      this.clearSession();
      window.location.href = "/login";
      return null;
    }
    if (!res.ok) {
      throw new Error(await friendlyError(res));
    }
    if (res.status === 204) return null;
    return res.json();
  },
};

function showToast(message) {
  let el = document.getElementById("toast");
  if (!el) {
    el = document.createElement("div");
    el.id = "toast";
    el.className = "toast";
    document.body.appendChild(el);
  }
  el.textContent = message;
  el.classList.add("show");
  clearTimeout(window.__toastTimer);
  window.__toastTimer = setTimeout(() => el.classList.remove("show"), 2800);
}

function initials(name) {
  if (!name) return "";
  return name.trim().split(/\s+/).filter(Boolean).slice(0, 2).map(p => p[0].toUpperCase()).join("");
}

function initTopbar() {
  const nameEl = document.getElementById("doc-name");
  const avatarEl = document.getElementById("doc-avatar");
  const doc = API.doctor();
  if (nameEl && doc) nameEl.textContent = `Dr. ${doc.full_name}`;
  if (avatarEl && doc) avatarEl.textContent = initials(doc.full_name);

  // "Previous Records" and the signed-in identity only make sense once a doctor is logged
  // in; "Login / Sign Up" only makes sense when they are not. Both classes are marked up on
  // every page's navbar, and toggled here from the same session state everything else reads.
  const loggedIn = !!API.token();
  document.querySelectorAll(".nav-auth-only").forEach(el => {
    el.style.display = loggedIn ? "" : "none";
  });
  document.querySelectorAll(".nav-guest-only").forEach(el => {
    el.style.display = loggedIn ? "none" : "";
  });

  const logoutBtn = document.getElementById("logout-btn");
  if (logoutBtn) logoutBtn.addEventListener("click", () => {
    API.clearSession();
    window.location.href = "/login";
  });

  const navToggle = document.getElementById("nav-toggle");
  const nav = document.getElementById("topbar-nav");
  if (navToggle && nav) {
    navToggle.addEventListener("click", () => {
      const open = nav.classList.toggle("show");
      navToggle.setAttribute("aria-expanded", open ? "true" : "false");
    });
    nav.querySelectorAll("a, button").forEach(el =>
      el.addEventListener("click", () => {
        nav.classList.remove("show");
        navToggle.setAttribute("aria-expanded", "false");
      }));
  }
}

function initFooter() {
  const yearEl = document.getElementById("foot-year");
  if (yearEl) yearEl.textContent = new Date().getFullYear();
}

// Fades/staggers .reveal-up elements in as they scroll into view. Elements not yet observed
// (e.g. no IntersectionObserver support) just render visible immediately rather than staying
// permanently hidden.
function initScrollReveal() {
  const items = document.querySelectorAll(".reveal-up");
  if (!items.length) return;
  if (!("IntersectionObserver" in window)) {
    items.forEach(el => el.classList.add("in-view"));
    return;
  }
  const io = new IntersectionObserver((entries) => {
    entries.forEach(entry => {
      if (entry.isIntersecting) {
        entry.target.classList.add("in-view");
        io.unobserve(entry.target);
      }
    });
  }, { threshold: 0.15, rootMargin: "0px 0px -40px 0px" });
  items.forEach(el => io.observe(el));
}
