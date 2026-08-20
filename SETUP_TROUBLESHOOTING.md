# Setup Troubleshooting

Notes on problems hit while setting up git/GitHub on this Mac, and what fixed them.

---

## `gh repo clone` fails with `failed to run git: exit status 129`

### Symptom

```
$ gh repo clone owner/repo -- --some-flag
   ... 40 lines of git usage text ...
failed to run git: exit status 129
```

### What it means

**129 is git's "bad usage" exit code.** git was handed an option it didn't
recognize, so it printed its usage block and quit. `gh` doesn't interpret the
number — it just relays whatever the git subprocess returned.

This is *not* an authentication, network, or SSH problem.

### Why the error looks cryptic

The actual explanation is the **first** line of output, but git's usage dump is
~40 lines long, so it scrolls off the top and only the useless `exit status 129`
line stays visible.

**Scroll up.** The real message is there:

```
error: unknown option `revision=main'
usage: git clone [<options>] [--] <repo> [<dir>]
```

Or pipe through `head` to see it directly:

```sh
gh repo clone owner/repo -- --some-flag 2>&1 | head -3
```

### Two causes

**1. A typo or genuinely unknown flag** after the `--` separator.

**2. A real flag that's too new for the installed git.** This is the one that
wastes time, because the flag is valid — you copied it from current docs — it
just doesn't exist in the older git actually being run. `git clone --revision`,
for example, requires git 2.49+.

macOS ships `/usr/bin/git` = **Apple Git 2.39.5**, which is years behind. That
was the cause here.

### The fix

Install a current git and make sure it wins on `PATH`:

```sh
brew install git
```

Then **open a new terminal** and confirm:

```sh
which git
git --version
```

You want `/opt/homebrew/bin/git` and version 2.5x — **not** `/usr/bin/git` at 2.39.x.

If `which git` still shows `/usr/bin/git`, `/opt/homebrew/bin` is not early
enough on `PATH`. Fix by prepending it in `~/.zshrc`:

```sh
export PATH="/opt/homebrew/bin:$PATH"
```

### Checking PATH correctly

A gotcha while diagnosing this: **`zsh -lc` does not source `~/.zshrc`.** A login
shell reads `.zshenv`, `.zprofile`, `.zlogin` — not `.zshrc`, which is for
*interactive* shells. Testing with `zsh -lc` reports a PATH your terminal never
actually uses, and it also inherits PATH from whatever process launched it.

To see what an interactive terminal really gets, use a clean environment:

```sh
env -i HOME="$HOME" TERM=xterm /bin/zsh -ic 'echo $PATH; which git'
```

### Related: 128 vs 129

Adjacent exit codes, completely different meanings:

| Code | Meaning | Example |
|------|---------|---------|
| **129** | Usage error — git rejected an *option* | `error: unknown option 'revision=main'` |
| **128** | Runtime error — options were fine, the *operation* failed | `fatal: Remote revision main not found in upstream origin` |

So if 129 turns into 128 after upgrading git, that's progress: the flag is now
understood, and you're looking at a real error. (In the example above, the repo's
default branch was `master`, not `main`.)

### Passing git flags through `gh`

`gh repo clone` needs `--` before any flags meant for `git clone`. Without it,
`gh` tries to parse them itself and prints its *own* usage instead:

Wrong — `gh` tries to parse `--depth` itself and prints its own usage:

```sh
gh repo clone owner/repo --depth=1
```

Right — everything after `--` is forwarded to `git clone`:

```sh
gh repo clone owner/repo -- --depth=1
```

---

## `fatal: Too many arguments.` when pasting a command into zsh

### Symptom

You copy a documented command that has an explanatory comment on the end:

```zsh
gh repo clone JBedichek/Local-arxiv-Research-Assistant    # private — see Access below
```

and get:

```
fatal: Too many arguments.

usage: git clone [<options>] [--] <repo> [<dir>]
```

### What it means

**zsh does not treat `#` as a comment in interactive shells.** The `interactive_comments`
option is **off by default** — this is a genuine difference from bash, where comments work at
the prompt without configuration.

So zsh does not discard the trailing note. It splits it into words and passes them along:

```
gh repo clone JBedichek/Local-arxiv-Research-Assistant '#' private '—' see Access below
```

`gh` forwards surplus positional arguments to `git clone`, which takes at most
`<repo> <dir>`. Eight operands is six too many, hence `fatal: Too many arguments.`

Check your own shell with:

```zsh
setopt | grep -i interactivecomments
```

No output means the option is **off** — which is the default, and the cause of this error.

### Two fixes

**Per-paste:** strip the comment. Run only the command itself.

**Permanent:** turn comments on, so pasted lines behave the way the docs assume.

```zsh
echo 'setopt interactive_comments' >> ~/.zshrc && exec zsh
```

### Tells that you pasted prose, not a command

- An **em-dash** (`—`) or other typographic punctuation. Terminals produce `--`, not `—`.
- Comment text that refers to the *document* rather than the command — "see Access below"
  is a cross-reference in a README, meaningless at a shell prompt.

### Related: 128 vs 129 again

This one is exit **128**, not 129. The distinction from the section above holds: the
*options* were all valid, so it is not a usage error — there were simply too many operands
for the command to act on.

