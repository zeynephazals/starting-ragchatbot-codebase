# Frontend Changes — Dark / Light Theme Toggle

Added a theme toggle that lets users switch between the existing dark theme and a
new light theme. The choice is persisted across page loads and the UI animates
smoothly when toggling.

## Files Changed

### `frontend/index.html`
- Added a fixed-position toggle button (`#themeToggle`) at the top of the
  `.container`, anchored to the **top-right** of the viewport.
- The button is icon-based, containing two inline SVGs: a **sun** icon
  (shown in light mode) and a **moon** icon (shown in dark mode). Only one is
  visible at a time; the other fades/rotates out.
- The button is a native `<button>` with an `aria-label` and `title`, so it is
  **accessible and keyboard-navigable** (Tab to focus, Enter/Space to activate).

### `frontend/style.css`
- **Theme variables**
  - Renamed the `:root` block to "Dark Theme (default)" and added several new
    custom properties so previously hard-coded colors can react to the theme:
    `--code-bg`, `--link-color`, `--link-hover`, `--source-bg`, `--source-border`,
    `--source-hover-bg`, `--source-hover-border`.
  - Added a `:root[data-theme="light"]` block defining the **light theme**:
    light backgrounds (`#f8fafc` / `#ffffff`), dark text (`#0f172a`) for strong
    contrast, adjusted secondary/border/surface colors, and theme-aware link,
    code, and source-pill colors. Primary blue is kept for brand consistency.
- **Smooth transitions** — added a shared `transition` on `background-color`,
  `color`, and `border-color` for the body and major surfaces so theme changes
  animate over 0.3s instead of snapping.
- **Toggle button styles** (`.theme-toggle`)
  - Fixed, top-right, 44×44 circular button using theme surface/border colors
    and the existing `--shadow`.
  - Hover lift + `:focus-visible` focus ring matching the app's existing
    `--focus-ring` style.
  - Icon crossfade animation: the sun/moon SVGs rotate and fade between states
    based on the `data-theme` attribute.
- **Color refactors** — replaced hard-coded hex/rgba values in `.sources-content a`,
  `.message-content a`, `.message-content code`, and `.message-content pre` with
  the new theme variables so they look correct in both themes.

### `frontend/script.js`
- Added `themeToggle` to the tracked DOM elements.
- `initTheme()` runs immediately on script load (before `DOMContentLoaded`) to
  apply the saved theme early and avoid a flash of the wrong theme.
- `applyTheme(theme)` sets/removes the `data-theme="light"` attribute on
  `<html>` and keeps the button's `aria-label`/`title` in sync with the action
  it will perform next.
- `toggleTheme()` flips the theme, applies it, and saves the choice to
  `localStorage` under the `theme` key.
- Wired the toggle's `click` handler in `setupEventListeners()`. Keyboard
  activation (Enter/Space) is handled natively because it's a `<button>`.

## Implementation Notes
- Theme switching is driven entirely by CSS custom properties + a single
  `data-theme` attribute on the `<html>` element, so all existing components
  inherit the correct colors with no per-element JS.
- Dark remains the default (no attribute); light is the explicit opt-in.
- The visual hierarchy, spacing, and design language are unchanged — only the
  color palette swaps.

## Verification
- Rendered the page in a browser and confirmed:
  - Default load shows the dark theme with the moon icon.
  - Clicking the toggle switches to the light theme (light background, dark
    text, sun icon) with a smooth transition.
  - The button is focusable and operable via keyboard.
