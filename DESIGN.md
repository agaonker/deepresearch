# Design System — deepresearch

## Product Context
- **What this is:** Local-only ReAct tool-calling research agent. CLI-first, runs on your machine, structured renders emitted by `render_*` tools.
- **Who it's for:** Builders running local agents. People who care that their tool reads as engineered, not packaged.
- **Space/industry:** Agent / dev-tooling CLIs (peers: `gh`, `claude` (Claude Code), `lazygit`, `k9s`, Charm `crush`).
- **Project type:** CLI / terminal UI. The design system governs terminal output emitted by `src/deepresearch/streaming/render_cli.py` painters.

## Memorable Thing
"It's a real agent, on your machine." Local-first, observable, no SaaS lock-in. Every aesthetic choice serves this — design says "tool," not "app."

## Aesthetic Direction
- **Direction:** Brutalist-precise. Engineered, exposed, no decoration. `htop`/`gh` discipline tightened with modern unicode box-drawing.
- **Decoration level:** Minimal. Structure does the work — box-drawing characters and four emphasis levels carry hierarchy.
- **Mood:** Disciplined. Information-dense. Looks at home in `tmux` next to compiler output, not in a marketing screenshot.
- **Reference points:** htop (raw clarity), gh CLI (typographic restraint), Charm crush (unicode discipline), lazygit (purposeful color).

## Color

### Palette
| Token   | Truecolor  | ANSI 256 | ANSI 16 fallback | Use                                       |
|---------|-----------|----------|------------------|-------------------------------------------|
| accent  | `#5FAFD7` | 74       | cyan (36)        | titles, headers, Q/A markers              |
| muted   | `#6C6C6C` | 8        | bright-black     | metadata, sources, separators, connectors |
| success | `#87D75F` | 113      | green (32)       | ok / done / passed                        |
| warn    | `#FFD75F` | 221      | yellow (33)      | partial / caveat                          |
| error   | `#FF5F5F` | 203      | red (31)         | failed / refused                          |
| bold    | (no hue)  | bold     | bold             | numeric emphasis in tables                |
| default | terminal  | —        | —                | body text                                 |
| bg      | terminal  | —        | —                | **NEVER override** — user's theme wins    |

### Color rules
- **Color never touches box-drawing.** Borders are structure (permanent). Color is signal (ephemeral, stripped on copy-paste). If you strip ANSI from any render, structure and hierarchy must remain legible.
- **Semantic only.** Color encodes meaning (status, role, emphasis), never decoration.
- **Auto-disable** when stdout is not a TTY OR `NO_COLOR` is set. Tests and pipes get monochrome automatically. See `streaming/render_tokens.set_color_enabled()`.
- **No painted backgrounds.** The user's iTerm/Alacritty/wezterm theme bleeds through. Local-first values applied to aesthetics.

## Typography

Fonts are fixed by the user's terminal — the design system can't pick them. Emphasis is built from four levels:

| Level   | ANSI         | Use                                                |
|---------|--------------|----------------------------------------------------|
| default | —            | body text                                          |
| muted   | dim + muted  | metadata, sources, connectors, secondary info      |
| bold    | bold         | numeric emphasis in tables                         |
| accent  | bold + accent| titles, headers, Q/A markers                       |

**Case is preserved as supplied.** Do not uppercase titles in painters — let callers control case. Bold + accent already shouts.

## Box-Drawing Tokens

Defined in `src/deepresearch/streaming/render_tokens.py` (`Box` class). Two frame weights:

### Standard (single-line, sharp)
```
┌─────────────┐
│ frame       │
├─────────────┤
│ separator   │
└─────────────┘
```
Use: `render_card` (default), `render_table`.

### Heavy (final committed answer)
```
┏━━━━━━━━━━━━━┓
┃ frame       ┃
┣━━━━━━━━━━━━━┫
┃ separator   ┃
┗━━━━━━━━━━━━━┛
```
Use: `render_card` when `heavy=True` is set. Reserved for the agent's terminal moment — the answer it commits to. Use sparingly. Multiple heavy frames in one session dilutes the meaning.

### Connectors
- Tree / timeline branch: `├──`
- Tree / timeline last: `└──`
- Tree continuation pipe: `│   ` (3 spaces)
- Tree empty indent: `    ` (4 spaces)

### Bar chart
- Solid bar: `█` (`chart_type == "bar"`)
- Dot scatter: `·` (`chart_type != "bar"`)
- No box around charts — boxes distort proportion read.

## Spacing
- **Base unit:** 1 character cell.
- **Inner padding:** 1 cell (`│ {content} │`).
- **Content width:** min 32, max 80. `render_card` and `render_table` both respect this.
- **Vertical:** 1 blank line between distinct renders. None within a single render.
- **Density:** Tight. Builders prefer information density over generous whitespace.

## Layout

### render_qa
```
Q  {question}

A  {answer}

sources
  · {source 1}
  · {source 2}
```
Lightest weight. Q/A markers in accent+bold. Body default. Sources prefixed with `· ` and dimmed.

### render_card
Sharp single-line box. Title (bold+accent) → separator → wrapped content → separator → metadata (key dimmed). When `heavy=True`, heavy box.

### render_table
Sharp single-line box. Title above (bold+accent). Header row bold+accent. Numeric columns auto-detected (every cell parses as a number, optionally with `$`, `%`, `,`, `.`, `-`, `+`, `/`) and rendered right-aligned + bold (no color).

### render_chart
No box. Label (muted, left-padded) → muted `│` → accent bar → bold value. Replaces `#` and `*` with `█` and `·`.

### render_timeline
Title (bold+accent). Then `├──` / `└──` (muted) + date (accent, padded) + label (default).

### render_tree
Title (bold+accent). Root (bold). Children via `├── ` / `└── ` (muted) with continuation pipe.

## Motion
Not applicable. Terminal output is streamed once; we don't re-render. If a render is emitted, it stays on screen — readable when scrolled back.

## Survives-Copy-Paste Rule
**The load-bearing constraint.** ANSI escape sequences are stripped when users paste terminal output into Slack, GitHub issues, Notion, or PR descriptions. So the design must remain legible without color:

- Hierarchy via box-drawing weight (sharp vs heavy) — color-independent.
- Numeric emphasis via bold (color-independent in most renderers that strip ANSI).
- Indentation and spacing — color-independent.
- Q/A and section markers — single-letter labels (`Q`, `A`, `sources`) work without color.

If you can paste a render into a markdown file and the structure still tells the story, the design is honest. Test by piping the CLI through `cat` or `pbcopy` and pasting back.

## Where This Lives in Code

| Concern              | File                                            |
|----------------------|-------------------------------------------------|
| Tokens (palette, box) | `src/deepresearch/streaming/render_tokens.py`  |
| Painters             | `src/deepresearch/streaming/render_cli.py`     |
| Render tool emitters | `src/deepresearch/tools/render.py`             |
| Render contract      | `src/deepresearch/streaming/events.py` (`_SENTINEL`) |
| Tests                | `tests/test_render.py`                          |

Adding a new render kind requires three coordinated edits (per CLAUDE.md): the `@tool` in `render.py`, a `_paint_<kind>` painter in `render_cli.py`, and a `_PAINTERS` registration. New painters must follow the conventions in this file.

## Decisions Log
| Date       | Decision                                                                  | Rationale                                                                                  |
|------------|---------------------------------------------------------------------------|--------------------------------------------------------------------------------------------|
| 2026-05-16 | Adopted brutalist-precise direction with single-line box-drawing          | Says "engineered tool" not "polished app" — matches "real agent, on your machine"          |
| 2026-05-16 | Color never touches box-drawing                                           | Output gets pasted into Slack/PRs; structure must survive ANSI stripping                   |
| 2026-05-16 | Reserved heavy frame (`┏━┓`) for final committed answer only              | Visual weight signals "this is the answer," not just structure                             |
| 2026-05-16 | Auto-detect TTY + respect `NO_COLOR`; tests are monochrome                | No flag plumbing required; pytest captures stdout so coloring auto-disables in tests       |
| 2026-05-16 | Numeric columns auto-detected and right-aligned + bold (no color)         | Numbers should jump in tables; color noise distracts; bold preserved on copy-paste         |
| 2026-05-16 | Case preserved in titles (no auto-uppercase)                              | Let callers control voice; bold+accent is enough emphasis                                  |
| 2026-05-16 | Terminal background never overridden                                      | Local-first respect: tool looks at home in the user's environment, not its own            |
