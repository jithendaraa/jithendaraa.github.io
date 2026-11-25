(function () {
  var themeButton = document.querySelector(".theme-toggle");
  function updateThemeButton() {
    var dark = document.documentElement.dataset.theme === "dark";
    themeButton.setAttribute("aria-label", dark ? "Switch to light theme" : "Switch to dark theme");
  }
  themeButton.hidden = false;
  updateThemeButton();
  themeButton.addEventListener("click", function () {
    var theme = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
    document.documentElement.dataset.theme = theme;
    updateThemeButton();
    try { localStorage.setItem("portfolio-theme", theme); } catch (error) { /* Storage is optional. */ }
  });
})();

(function () {
  const note = document.querySelector(".astra-note");
  if (!note) return;
  const credit = note.querySelector("summary");
  let preview = false;

  function closeNote() {
    preview = false;
    note.open = false;
  }

  note.addEventListener("pointerenter", event => {
    if (event.pointerType !== "mouse" || note.open) return;
    preview = true;
    note.open = true;
  });
  note.addEventListener("pointerleave", () => {
    if (preview && !note.contains(document.activeElement)) closeNote();
  });
  credit.addEventListener("click", event => {
    if (!preview) return;
    event.preventDefault();
    preview = false;
  });
  document.addEventListener("pointerdown", event => {
    if (note.open && !note.contains(event.target)) closeNote();
  });
  note.addEventListener("focusout", event => {
    if (!note.contains(event.relatedTarget)) closeNote();
  });
  document.addEventListener("keydown", event => {
    if (event.key !== "Escape" || !note.open) return;
    if (note.contains(document.activeElement)) credit.focus();
    closeNote();
  });
})();

(function () {
  const gallery = document.querySelector(".video-gallery");
  if (!gallery) return;

  const strip = gallery.querySelector(".video-strip");
  const frames = Array.from(strip.children);
  if (frames.length < 2 || !strip.animate) return;

  // Each frame wraps only after it has cleared the left edge. Keeping the same
  // video element lets its footage keep playing across the wrap.
  const step = 9000;
  const route = [
    { transform: "translateX(" + (frames.length - 1) * 100 + "%)" },
    { transform: "translateX(-100%)" }
  ];
  const animations = frames.map((frame, index) => frame.animate(route, {
    duration: frames.length * step,
    delay: -(frames.length - 1 - index) * step,
    iterations: Infinity,
    easing: "linear"
  }));

  const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)");
  function syncMotion() {
    strip.classList.toggle("is-moving", !reducedMotion.matches);
    for (const animation of animations) {
      if (reducedMotion.matches) animation.cancel();
      else animation.play();
    }
  }
  syncMotion();
  reducedMotion.addEventListener("change", syncMotion);
})();
