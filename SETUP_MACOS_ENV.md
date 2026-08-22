# macOS Bootstrap

> **Copyright © 2026 James Bedichek. All rights reserved.** This is proprietary software,
> not open source. These instructions describe how to run it; they do not grant permission
> to use it. Copying, modifying, redistributing, or commercial use require prior written
> permission — see [`LICENSE`](LICENSE).

Getting a **factory-fresh Mac** to the point where [`SETUP_INSTRUCTIONS.md`](SETUP_INSTRUCTIONS.md)
can start. That file assumes you already have `git`, a package manager, and Python 3.12+.
A new Mac has **none of them**, and the way it fails to have them is misleading enough to
cost an hour. This file covers only that gap, and stops at the clone.

**Scope boundary.** Nothing here duplicates `SETUP_INSTRUCTIONS.md`. The venv, `pip install`,
Hugging Face auth, the corpus pull and the wizard all live there — do **not** run them from
here. When you reach [Hand-off](#hand-off), switch files.

**Every block on this page is comment-free and safe to paste whole** — zsh does not strip
`#` comments interactively. See [`SETUP_TROUBLESHOOTING.md`](SETUP_TROUBLESHOOTING.md).

---

## The whole thing

Six steps. **Three of them stop and wait for you** — a dialog to click (Command Line Tools),
a password to type (Homebrew), and a browser to approve (`gh auth login`) — so this is not a
paste-and-walk-away block. Read [The steps](#the-steps) for what each one does and which
traps it avoids.

```zsh
xcode-select --install

sudo -v
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
eval "$(/opt/homebrew/bin/brew shellenv)"

brew install python@3.12
brew install llama.cpp
brew install gh

git config --global user.name "Your Name"
git config --global user.email "you@example.com"

gh auth login --git-protocol https --web
gh auth status

cd ~/Desktop
gh repo clone JBedichek/Local-arxiv-Research-Assistant
```

| step | why it is here |
|---|---|
| `xcode-select --install` | **Nothing else works first.** `git` and `python3` are stubs until this lands |
| Homebrew | the only way to get Python 3.12 and `llama.cpp`; macOS ships neither |
| `brew install python@3.12` | system Python is **3.9.6**; this project requires **≥3.12** |
| `brew install llama.cpp` | **the default generator on Apple Silicon** — a binary, so pip cannot install it |
| `brew install gh` | **`SETUP_INSTRUCTIONS.md` opens with `gh repo clone`** — that command does not exist on a new Mac |
| `git config --global` | git refuses to commit without an identity |
| `gh auth login` | credentials for pushing. Prefer keys? [Step 6](#6-github-access) has the SSH route |

---

## The steps

### 1. Xcode Command Line Tools

**This is the step that confuses people, so do it first and do not skip it.** A new Mac
*appears* to have git and Python:

```zsh
which git
which python3
```

Both answer — `/usr/bin/git` and `/usr/bin/python3`. **Both are stubs.** They are shims that
do nothing but trigger the developer-tools installer. Running *any* of them, including
`git --version`, pops a GUI dialog and prints:

```
xcode-select: note: No developer tools were found, requesting install.
```

Install them:

```zsh
xcode-select --install
```

**A GUI dialog opens — you must click "Install" and accept the licence.** It is ~920 MB and
takes several minutes. The command returns immediately; that does not mean it is done.

Confirm it actually finished:

```zsh
xcode-select -p
git --version
```

You want `/Library/Developer/CommandLineTools` and a real version string. Until then, every
command in every later step fails with the `xcode-select: note:` message above.

> **`sudo softwareupdate --install "Command Line Tools for Xcode 26.6"` is the headless
> alternative**, useful over SSH where no one can click a dialog. Get the exact label from
> `softwareupdate --list`; it changes with every OS point release.

### 2. Homebrew

```zsh
sudo -v
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

**Run this in Terminal.app, not through a tool or an editor's shell pane.** It needs a real
TTY to prompt for your password.

> **The trap: `NONINTERACTIVE=1` produces a flatly wrong error message.** That variable makes
> the installer use `sudo -n` (non-interactive), which fails the instant your password is not
> already cached. Homebrew reports that failure as:
>
> ```
> Need sudo access on macOS (e.g. the user <you> needs to be an Administrator)!
> ```
>
> **This does not mean your account lacks admin rights.** It almost always has them — the
> first account on a Mac is an administrator by default. Check before believing it:
>
> ```zsh
> dscl . -read /Groups/admin GroupMembership
> ```
>
> If your username is in that list, you are an admin and the message is a red herring.
> `sudo -v` on the line before caches your credentials and makes it go away. Only use
> `NONINTERACTIVE=1` in genuine automation, and only after `sudo -v`.

#### PATH

The installer appends this to `~/.zprofile` for you:

```zsh
eval "$(/opt/homebrew/bin/brew shellenv zsh)"
```

That is enough for new terminal windows, but **your current shell has not read it**. Either
open a new terminal or run the same line by hand before continuing.

> **`~/.zprofile` is read by login shells only.** Scripts, `zsh -c`, and tool-driven shells
> are not login shells, so `brew` will appear missing inside them even though it is installed
> and works fine in your terminal. In a script, call `/opt/homebrew/bin/brew` by absolute
> path or `eval "$(/opt/homebrew/bin/brew shellenv)"` first. This is the mirror image of the
> `zsh -lc` / `~/.zshrc` gotcha already documented in
> [`SETUP_TROUBLESHOOTING.md`](SETUP_TROUBLESHOOTING.md).

Confirm:

```zsh
brew --version
```

### 3. Python 3.12

```zsh
brew install python@3.12
```

**macOS ships Python 3.9.6 and this project requires 3.12 or newer** (`requires-python =
">=3.12"`). 3.9 is not close enough to limp along on — the install fails outright.

```zsh
/usr/bin/python3 --version
python3.12 --version
```

The first prints `3.9.6`, the second `3.12.x`. Both are correct and both stay on the machine.

> **Use `python3.12` by name when you create the venv, never bare `python3`.** Homebrew
> deliberately does not make `python@3.12` the default `python3`, so bare `python3` stays
> the system 3.9.6. `SETUP_INSTRUCTIONS.md` writes `python3 -m venv .venv`; on a fresh Mac,
> prefer:
>
> ```zsh
> python3.12 -m venv .venv
> ```
>
> **The resulting error names neither Python nor its version**, which is what makes this
> cost an hour. `pip install -e` fails with `File "setup.py" or "setup.cfg" not found …
> editable mode currently requires a setuptools-based build` — an accusation against the
> project. The real cause is that Python 3.9 ships pip 21.2.4, which predates PEP 660 and so
> cannot install a hatchling project in editable mode. Confirm with `python --version` and
> `cat .venv/pyvenv.cfg` rather than trusting the message, and rebuild rather than upgrading
> pip inside a 3.9 venv.
>
> **`deactivate` before creating it.** Inside an active 3.9 venv, `python3` is that venv's
> 3.9, so the new venv silently inherits it — `pyvenv.cfg` records which interpreter was
> actually used, and is the fastest way to catch this.

**3.12 rather than 3.13 or newer** is deliberate: it is the version the ML wheel stack
(torch, sentence-transformers, mlx-lm) has the most complete coverage for. Newer usually
works; 3.12 is the one that reliably does.

### 4. llama.cpp

```zsh
brew install llama.cpp
```

Installs `llama-server` and `llama-cli`. It is a **compiled binary, not a Python package**,
which is why it cannot come in through `pip` with the rest and has to be handled at this
layer. `SETUP_INSTRUCTIONS.md` refers to `brew install llama.cpp` and assumes Homebrew is
already present — step 2 is what makes that true.

**This is the default generator on Apple Silicon, and the one component pip cannot supply.**
Without it the backend resolver falls back to MLX — which does work, since `mlx-lm` ships
with `pip install -e '.[mac]'` — but you get a runtime the documentation does not describe,
reading a different weight format, chosen because it was the thing that installed itself.
Availability is decided purely by whether `llama-server` is on PATH.

MLX is often faster on this hardware, so having both and benchmarking them is worth it —
the repo documents `lara backends` and `lara bench-generate` for exactly that comparison.

### 5. git identity

```zsh
git config --global user.name "Your Name"
git config --global user.email "you@example.com"
git config --global init.defaultBranch main
```

A fresh Mac has no git identity, and git **refuses to create a commit** without one. It does
not block cloning, so the failure surfaces later, at your first commit.

> **You very likely do not need `brew install git`.** The version the Command Line Tools give
> you depends on your macOS release: 2.39.5 on Ventura through early Sequoia, but **2.50.1 on
> macOS 26** — current enough for `git clone --revision` and everything else. Check before
> installing a second git:
>
> ```zsh
> git --version
> ```
>
> At 2.49 or newer, skip it. Below that, see
> [`SETUP_TROUBLESHOOTING.md`](SETUP_TROUBLESHOOTING.md) — and note its warning that
> `/opt/homebrew/bin` must come first on `PATH` for the new git to actually win.

### 6. GitHub access

**Pick one of the two routes below — you do not need both.** Cloning itself needs no
credentials while this repository is public; what you are setting up here is the ability to
**push**.

#### Route A — GitHub CLI (matches `SETUP_INSTRUCTIONS.md`)

```zsh
brew install gh
gh auth login --git-protocol https --web
gh auth status
```

**Install `gh` even if you choose Route B.** Every clone line in `SETUP_INSTRUCTIONS.md` is
written as `gh repo clone`, and on a new Mac that command does not exist — the file's very
first step fails with `command not found: gh`. One `brew install gh` makes the documented
path work as written.

> **`--git-protocol https` is load-bearing.** Choosing `ssh` at that prompt leaves git itself
> without a credential helper, and HTTPS clones then fail with `could not read Username`
> even though `gh` says you are logged in. `gh auth setup-git` repairs it after the fact.
> `SETUP_INSTRUCTIONS.md` documents this trap in full.

Browser-less machine? `gh auth login --with-token < token.txt`.

#### Route B — SSH key

Use this if you prefer keys, or want the `git@github.com:` clone URL. A new Mac has no
`~/.ssh` at all, so that URL cannot authenticate until you make one.

```zsh
ssh-keygen -t ed25519 -C "you@example.com" -f ~/.ssh/id_ed25519 -N ""
```

`-N ""` sets an empty passphrase. For a passphrase-protected key, drop `-N ""` and type one
when prompted; the `--apple-use-keychain` flag below then stores it so you are asked once
rather than every push.

Make the agent load it automatically on every reboot:

```zsh
cat > ~/.ssh/config <<'EOF'
Host github.com
  HostName github.com
  User git
  IdentityFile ~/.ssh/id_ed25519
  AddKeysToAgent yes
  UseKeychain yes
EOF
chmod 600 ~/.ssh/config
ssh-add --apple-use-keychain ~/.ssh/id_ed25519
```

`UseKeychain` and `--apple-use-keychain` are macOS-specific and are what stop the key from
being forgotten on restart.

Register the public key with GitHub:

```zsh
pbcopy < ~/.ssh/id_ed25519.pub
open "https://github.com/settings/ssh/new"
```

Paste, give it a title identifying the machine, and save. **Copy `id_ed25519.pub`, never
`id_ed25519`** — the file without the extension is your private key and must not leave the
machine.

Verify before trying to clone:

```zsh
ssh -T git@github.com
```

Success looks like this, and the "does not provide shell access" half is **not** an error:

```
Hi <username>! You've successfully authenticated, but GitHub does not provide shell access.
```

> **`ssh-keygen -h` does not print help.** In `ssh-keygen`, `-h` means "host key" — it starts
> generating a key and drops you at an interactive prompt with no explanation. Use
> `man ssh-keygen`.

> **This replaces Route A's `gh auth login`, not `brew install gh`.** Authenticating twice is
> redundant, but the `gh` *binary* is still worth having, because `SETUP_INSTRUCTIONS.md`
> spells every clone as `gh repo clone`. With a key configured you can equally substitute
> `git clone git@github.com:…` wherever that file says `gh repo clone`.

---

## Hand-off

```zsh
cd ~/Desktop
gh repo clone JBedichek/Local-arxiv-Research-Assistant
cd Local-arxiv-Research-Assistant
```

With an SSH key instead of `gh`, the equivalent is:

```zsh
git clone git@github.com:JBedichek/Local-arxiv-Research-Assistant.git
```

**Stop here and switch to [`SETUP_INSTRUCTIONS.md`](SETUP_INSTRUCTIONS.md)**, at
[step 2, Install](SETUP_INSTRUCTIONS.md#2-install). It takes over from the venv onward —
`pip install -e '.[mac]'`, `hf auth login`, `lara dataset pull`, `lara setup`, `lara serve`.
One deviation from what it prints, for the reason in step 3 above: build the venv with
`python3.12 -m venv .venv`, not `python3 -m venv .venv`.

The clone command is repeated here only because you cannot read a file in a repository you
have not cloned.

---

## Verify the bootstrap

Everything below should answer before you switch files. Any `MISSING` or a Python under 3.12
means a step above did not finish.

```zsh
xcode-select -p
git --version
brew --version
python3.12 --version
llama-server --version
gh --version
gh auth status
ssh -T git@github.com
```

The last two lines are route-dependent: `gh auth status` is the check for Route A,
`ssh -T git@github.com` for Route B. Whichever you skipped will report "not logged in" or
"Permission denied", and that is expected rather than a failure.

Reference output from a verified run of this document, macOS 26.5.1 on Apple Silicon:

| tool | expected | verified |
|---|---|---|
| Command Line Tools | a path | `/Library/Developer/CommandLineTools` |
| git | ≥ 2.39, ideally ≥ 2.49 | `2.50.1 (Apple Git-155)` |
| Homebrew | any current | `6.0.18` |
| Python (brew) | ≥ 3.12 | `3.12.14` |
| Python (system) | ignored, stays 3.9 | `3.9.6` |
| llama.cpp | any current | `build 10520` |
| GitHub CLI | any current | route-dependent |
| GitHub SSH | `Hi <username>!` | authenticated |

---

## What this file does not cover

| not here | where |
|---|---|
| venv, `pip install -e '.[mac]'` | [`SETUP_INSTRUCTIONS.md` §2](SETUP_INSTRUCTIONS.md#2-install) |
| Hugging Face account, gated embedder, `hf auth login` | [`SETUP_INSTRUCTIONS.md`](SETUP_INSTRUCTIONS.md#getting-access-do-this-first) |
| corpus download, `lara setup`, `lara serve` | [`SETUP_INSTRUCTIONS.md` §3–5](SETUP_INSTRUCTIONS.md#3-download-the-corpus) |
| zsh `#` comments, `gh` exit 129, PATH vs `zsh -lc` | [`SETUP_TROUBLESHOOTING.md`](SETUP_TROUBLESHOOTING.md) |
| choosing a search backend or scoping a small corpus | [`SETUP_INSTRUCTIONS.md` §4](SETUP_INSTRUCTIONS.md#4-run-the-wizard) |

**Disk and memory are checked by `lara setup`, not here** — but budget **~50 GB** for the
`core` corpus before you start, and note that a Mac under ~24 GB of RAM will be advised to
trim the corpus. A 48 GB machine runs it untouched.
