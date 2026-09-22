# Knossos Field — what it is and how it should feel

Status: the product statement. Every interface decision is measured against this
document; when a pass and this document disagree, one of them is wrong and it is
worth saying which before writing code.

## The one sentence

Every other agent tool is a chat log. Knossos gives your agents a map: you see
where they are working, what ground nobody has touched, what needs your
decision, and you watch a project visibly grow as verified work lands.

## The two halves

**The strategy game is the interface.** A project is a settlement. The folders
inside it are the districts around it. An agent is a unit standing in the folder
whose file it has open. Territory nobody has touched goes dark. Time runs along
the bottom of the map and you can drag it backwards to watch the work happen.

**The harness is the substance.** Children run inside a workspace jail on Linux
and macOS. Consequential actions stop at a permission gate. Money is reserved
before work starts and the session stops when it runs out. An independent
verifier decides whether a change is done, not the model's own claim.

Neither half is decoration. The map exists because an operator running six
agents across two repositories cannot hold that in their head from a list, and
a map is the oldest answer to that problem. The harness exists because an agent
you cannot stop, bound or check is a liability whatever it looks like.

## The rule that keeps it honest

**The game layer may never invent signal.**

A settlement grows because verified work landed in it. A district darkens
because nobody has touched it in a week. A unit stands in `src/field` because
that is the file it has open right now. Crowding, heat, staleness, growth and
rank are all read from the event log, and the event log is append-only fact.

No experience points. No levels. No streaks, no daily rewards, no progress bar
that fills because time passed. If a thing moves on the map, something real
moved underneath it. That constraint is not a tax on the fun; it *is* the fun.
A world that moves for no reason is a screensaver.

Corollary: we do not gamify the operator either. There is no score for
approving faster, and nothing nags. The app is quiet until something needs a
person.

## The loop

1. **Look.** Open Rome. The world shows you where work is happening and where
   it is not.
2. **Notice.** Something wants you: an agent is blocked, a permission is
   pending, a change is ready to accept. It is visible from the world, not
   buried in a panel.
3. **Tap.** The district opens. You see the conversations happening there.
4. **Decide.** One card, the context that matters, one or two buttons. Approve
   the action, redirect the agent, or accept the change.
5. **Watch.** Work progresses on the map, in the open.
6. **Collect.** Accepting a change is the beat: the diff lands, the district
   registers it, and over time the settlement grows.
7. **Send the next one.** Start an agent on a folder and go back to step one.

Everything else in the product is in service of that loop or is one tap deeper
than it.

## What the first second may contain

An operator glancing at the screen gets: where the work is, what needs them,
and how much is running. That is three things. Anything else — elapsed time,
context percentage, token counts, tool tallies, verification tiers, delegation
trees, per-session budgets — is real and useful and belongs exactly one
interaction away.

The console habit is to show every number at once because every number is
available. A game shows one number and hides the rest behind a tap. We are
building the second thing.

## Register and vocabulary

The interface speaks plainly. The world is illustrated, the words are not
decorated. "Needs approval", not "awaiting senatorial assent".

Two registers, kept apart on purpose:

- **World nouns** name places and things you can see and click: settlement,
  district, capital, unit, territory, map, replay.
- **Plain nouns** name what the operator is actually doing: agent,
  conversation, project, folder, file, change, approval, model, plan, routine,
  budget, verification.

Where both would work, the plain noun wins in buttons, labels and messages; the
world noun survives where it names something visual and unambiguous. We do not
invent a third register, and API names, event kinds and documentation stay
literal: `session`, `workspace`, `endpoint`, `permission`, `campaign`.

Retired for good: red team, blue team, referee, findings, mitigations,
verdicts-as-ritual, the senate, rehearsals, and character names for agents. An
agent is a configuration, not a persona.

## What an agent is

A model, a set of tools, a permission posture, whatever capabilities it has,
its standing instructions, a delegation limit, a budget and a definition of
done. Personality is which model it runs on. Roles are editable presets over
those settings, not character classes.

Not a swarm. One agent per task, delegating when a task is wide, run in
parallel across tasks rather than across personalities.

## How we know it is working

- A new operator opens the app, mounts a project, starts an agent and accepts
  its first change without reading documentation.
- An operator running six agents can say, from one glance, which one needs
  them.
- Nothing on the map is explainable only by "the interface felt like it".
- The app is silent when nothing needs a person.
