/* MediScan AI — UI behaviour (no inline scripts; CSP script-src 'self'). */
(function () {
  "use strict";

  /* ------------------------------------------------------------- theme */
  var THEME_KEY = "mediscan-theme";
  var root = document.documentElement;
  try {
    var saved = window.localStorage.getItem(THEME_KEY);
    if (saved === "light" || saved === "dark") {
      root.setAttribute("data-theme", saved);
    } else if (window.matchMedia &&
               window.matchMedia("(prefers-color-scheme: light)").matches) {
      root.setAttribute("data-theme", "light");
    }
  } catch (error) { /* private mode */ }

  function toggleTheme() {
    var next = root.getAttribute("data-theme") === "light" ? "dark" : "light";
    root.setAttribute("data-theme", next);
    try { window.localStorage.setItem(THEME_KEY, next); } catch (e) {}
    syncThemeButtons();
  }

  function syncThemeButtons() {
    var light = root.getAttribute("data-theme") === "light";
    document.querySelectorAll("[data-theme-toggle]").forEach(function (btn) {
      btn.setAttribute("aria-pressed", light ? "true" : "false");
      btn.setAttribute("title", light ? "Switch to dark theme"
                                     : "Switch to light theme");
    });
  }

  document.addEventListener("click", function (event) {
    var themeButton = event.target.closest("[data-theme-toggle]");
    if (themeButton) { toggleTheme(); return; }

    var printButton = event.target.closest("[data-print]");
    if (printButton) { window.print(); return; }

    var navToggle = event.target.closest("[data-nav-toggle]");
    if (navToggle) { toggleNav(); return; }

    if (event.target.closest(".nav-scrim")) { closeNav(); return; }

    var navLink = event.target.closest(".sidebar a");
    if (navLink) { closeNav(); }
  });

  document.addEventListener("keydown", function (event) {
    if (event.key === "Escape") { closeNav(); }
  });

  /* --------------------------------------------------------- mobile nav */
  var scrim = null;
  function toggleNav() { document.body.classList.contains("nav-open")
                         ? closeNav() : openNav(); }
  function openNav() {
    document.body.classList.add("nav-open");
    if (!scrim) {
      scrim = document.createElement("div");
      scrim.className = "nav-scrim";
      document.body.appendChild(scrim);
    }
  }
  function closeNav() {
    document.body.classList.remove("nav-open");
    if (scrim && scrim.parentNode) { scrim.parentNode.removeChild(scrim); }
    scrim = null;
  }

  /* --------------------------------------------------- busy form handling */
  function resetBusy(form) {
    delete form.dataset.submitting;
    form.querySelectorAll("button[data-original-text]").forEach(function (btn) {
      btn.textContent = btn.dataset.originalText;
      delete btn.dataset.originalText;
      btn.classList.remove("is-busy");
    });
  }

  document.addEventListener("submit", function (event) {
    var form = event.target;
    if (!(form instanceof HTMLFormElement)) { return; }
    if (form.dataset.confirm &&
        !window.confirm(form.dataset.confirm)) {
      event.preventDefault();
      return;
    }
    if (!form.dataset.busy) { return; }
    if (form.dataset.submitting === "1") {
      event.preventDefault();
      return;
    }
    form.dataset.submitting = "1";
    var button = event.submitter;
    if (!button || button.type !== "submit") {
      button = form.querySelector('button[type="submit"]');
    }
    if (button) {
      button.dataset.originalText = button.textContent;
      button.textContent = form.dataset.busy;
      button.classList.add("is-busy");
    }
  }, true);

  window.addEventListener("pageshow", function () {
    document.querySelectorAll("form[data-busy]").forEach(resetBusy);
  });

  /* -------------------------------------------------------------- dropzones */
  document.querySelectorAll("[data-dropzone]").forEach(function (zone) {
    var input = zone.querySelector('input[type="file"]');
    var label = zone.querySelector("[data-file-name]");
    if (!input) { return; }
    var hint = label ? label.dataset.defaultText || label.textContent : "";

    function showName() {
      if (!label) { return; }
      label.textContent = input.files && input.files.length
        ? input.files[0].name : hint;
    }

    ["dragenter", "dragover"].forEach(function (name) {
      zone.addEventListener(name, function (event) {
        event.preventDefault();
        zone.classList.add("dragover");
      });
    });
    ["dragleave", "dragend", "drop"].forEach(function (name) {
      zone.addEventListener(name, function (event) {
        event.preventDefault();
        zone.classList.remove("dragover");
      });
    });
    zone.addEventListener("drop", function (event) {
      if (event.dataTransfer && event.dataTransfer.files.length) {
        input.files = event.dataTransfer.files;
        input.dispatchEvent(new Event("change", { bubbles: true }));
      }
    });
    input.addEventListener("change", showName);
    showName();
  });

  /* ------------------------------------------------------- heatmap opacity */
  document.querySelectorAll("input[data-heatmap-range]").forEach(
    function (range) {
      var scope = range.closest("[data-heatmap-scope]") || document;
      var overlay = scope.querySelector("img[data-heatmap-overlay]");
      var output = scope.querySelector("[data-heatmap-value]");
      function update() {
        var value = Number(range.value);
        if (overlay) { overlay.style.opacity = String(value / 100); }
        if (output) { output.textContent = value + "%"; }
      }
      range.addEventListener("input", update);
      update();
    });

  /* ------------------------------------------------------------- lab rows */
  var labConfigNode = document.getElementById("lab-config");
  var labForm = document.getElementById("lab-form");
  if (labConfigNode && labForm) {
    var rangeData = {};
    try { rangeData = JSON.parse(labConfigNode.textContent); } catch (e) {}
    var tbody = labForm.querySelector("#lab-rows tbody");

    function fillRanges(row) {
      var testSelect = row.querySelector(".test-select");
      var rangeSelect = row.querySelector(".range-select");
      if (!testSelect || !rangeSelect) { return; }
      var wanted = row.getAttribute("data-range") || rangeSelect.value;
      var spec = rangeData[testSelect.value] || {};
      var keys = spec.ranges ? Object.keys(spec.ranges) : [];
      rangeSelect.textContent = "";
      keys.forEach(function (key) {
        var option = document.createElement("option");
        option.value = key;
        option.textContent = key;
        rangeSelect.appendChild(option);
      });
      if (keys.indexOf(wanted) !== -1) { rangeSelect.value = wanted; }
    }

    if (tbody) {
      tbody.querySelectorAll("tr").forEach(fillRanges);
      labForm.addEventListener("change", function (event) {
        if (event.target.classList.contains("test-select")) {
          var row = event.target.closest("tr");
          if (row) { fillRanges(row); }
        }
      });
      var addRow = document.getElementById("add-row");
      if (addRow) {
        addRow.addEventListener("click", function () {
          var first = tbody.querySelector("tr");
          if (!first) { return; }
          var row = first.cloneNode(true);
          row.removeAttribute("data-range");
          var valueInput = row.querySelector('input[name="value"]');
          if (valueInput) { valueInput.value = ""; }
          tbody.appendChild(row);
          fillRanges(row);
        });
      }
      labForm.addEventListener("click", function (event) {
        if (event.target.closest(".remove-row")) {
          var rows = tbody.querySelectorAll("tr");
          if (rows.length > 1) {
            event.target.closest("tr").remove();
          }
        }
      });
    }
  }

  syncThemeButtons();
})();
