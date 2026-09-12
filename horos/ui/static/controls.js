// horos shared form controls (see controls.css). Wires every static
// .stepper's chevron buttons: each click moves the input by its step,
// clamps to min/max, rounds to the step's decimals and fires input+change so
// page code reacts exactly as to typing. Pages that render steppers
// dynamically call horosControls.wire(root) after rendering.
(function () {
  const CHEVRON_UP = '<svg viewBox="0 0 10 10" fill="none" stroke="currentColor" '
    + 'stroke-width="1.6" stroke-linecap="round"><polyline points="2,6.5 5,3.5 8,6.5"/></svg>';
  const CHEVRON_DOWN = '<svg viewBox="0 0 10 10" fill="none" stroke="currentColor" '
    + 'stroke-width="1.6" stroke-linecap="round"><polyline points="2,3.5 5,6.5 8,3.5"/></svg>';

  function decimals(step) {
    const text = String(step);
    return text.includes(".") ? text.split(".")[1].length : 0;
  }

  function wire(root) {
    (root || document).querySelectorAll(".stepper button[data-step]").forEach((btn) => {
      if (btn.dataset.wired) return;
      btn.dataset.wired = "1";
      btn.tabIndex = -1;
      if (!btn.innerHTML.trim()) btn.innerHTML = btn.dataset.step === "1" ? CHEVRON_UP : CHEVRON_DOWN;
      btn.addEventListener("click", () => {
        const input = btn.dataset.for
          ? document.getElementById(btn.dataset.for)
          : btn.closest(".stepper").querySelector("input[type=number]");
        if (!input || input.disabled) return;
        const step = Number(input.step) || 1;
        let v = (Number(input.value) || 0) + step * Number(btn.dataset.step);
        if (input.min !== "") v = Math.max(Number(input.min), v);
        if (input.max !== "") v = Math.min(Number(input.max), v);
        input.value = Number(v.toFixed(Math.max(decimals(step), 6)));
        input.dispatchEvent(new Event("input", { bubbles: true }));
        input.dispatchEvent(new Event("change", { bubbles: true }));
      });
    });
    // file pickers: echo the chosen name next to the button when asked to
    (root || document).querySelectorAll(".file-btn input[type=file][data-echo]").forEach((inp) => {
      if (inp.dataset.wired) return;
      inp.dataset.wired = "1";
      inp.addEventListener("change", () => {
        const target = document.getElementById(inp.dataset.echo);
        if (target && inp.files && inp.files.length) target.textContent = inp.files[0].name;
      });
    });
  }

  // markup for a stepper around a number input; attrs is a string of extra attributes
  function stepper(id, value, { min = "", max = "", step = 1, cls = "", attrs = "" } = {}) {
    return '<span class="stepper ' + cls + '">'
      + '<input type="number" id="' + id + '" value="' + value + '" step="' + step + '"'
      + (min !== "" ? ' min="' + min + '"' : "") + (max !== "" ? ' max="' + max + '"' : "")
      + " " + attrs + ">"
      + '<span class="step-btns"><button type="button" data-step="1" title="+' + step + '"></button>'
      + '<button type="button" data-step="-1" title="−' + step + '"></button></span></span>';
  }

  window.horosControls = { wire, stepper, CHEVRON_UP, CHEVRON_DOWN };
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => wire());
  else wire();
})();
