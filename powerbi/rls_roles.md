# Row-level security

Two roles: a **CCR** sees only their own contacts; a **Team Lead** sees their whole
team. Both are dynamic — one role definition serves every user, matched at query
time on the signed-in identity.

All data is **synthetic** (see `CLAUDE.md`). The email addresses below are
generated on a `.example.com` domain, which RFC 2606 reserves and guarantees can
never resolve or receive mail — so no synthetic agent can collide with a real
mailbox.

---

## The identity columns

`src/build_model_tables.py` generates two email columns specifically for this:

| Table | Column | Example |
|---|---|---|
| `dim_agent` | `agent_email` | `robert.johnson@prairieskyutilities.example.com` |
| `dim_team` | `team_lead_email` | `danielle.johnson@prairieskyutilities.example.com` |

Both are built as `firstname.lastname@domain`, lowercased, with a numeric suffix
on any collision. A duplicate would hand one agent another agent's contacts, so
uniqueness is asserted in `tests/test_model_tables.py`.

> **In production** you would not generate these. You would import the real UPN
> from Entra ID / the workforce system and join on that. The rules below are
> unchanged either way — only the source of the column differs.

---

## Role 1 — CCR

**Intent:** an agent sees their own contacts and nothing else. No colleague's
calls, no team totals that would let them infer one.

**Table:** `dim_agent`  **Filter:**

```dax
[agent_email] = USERPRINCIPALNAME()
```

That single filter is enough for the whole model. `dim_agent` is on the one side
of a one-to-many relationship to `fact_interactions`, and filters propagate down
the chain, so restricting `dim_agent` to a single row automatically restricts
every contact, case, survey response and coaching note belonging to anyone else.

**What the CCR still sees, and why it is intentional:**

| Table | Filtered? | Effect |
|---|---|---|
| `dim_agent` | ✅ to 1 row | own record only |
| `fact_interactions` | ✅ via relationship | own contacts only |
| `dim_team` | ❌ | team names remain visible — it is a lookup, not data |
| `fact_sqm_monthly` | ❌ | **see the warning below** |
| `dim_date`, `dim_topic`, `dim_channel` | ❌ | pure lookups, no personal data |

> ⚠️ **`fact_sqm_monthly` is NOT filtered by the CCR role.** The scorecard is at
> month × team grain and hangs off `dim_team`, not `dim_agent`, so nothing in the
> CCR filter touches it. An agent would see every team's FCR.
>
> That is only acceptable if team-level FCR is considered non-sensitive. If it is
> not, add a second filter to the CCR role on `dim_team`:
>
> ```dax
> [team_name] IN
>     CALCULATETABLE (
>         VALUES ( dim_team[team_name] ),
>         FILTER ( ALL ( dim_agent ), dim_agent[agent_email] = USERPRINCIPALNAME() )
>     )
> ```
>
> This is the single easiest RLS mistake to make in this model: securing the fact
> you were thinking about and leaving the one you were not.

---

## Role 2 — Team Lead

**Intent:** a lead sees every agent on the team they lead, including themselves if
they also handle contacts. Leads do not see other teams.

**Table:** `dim_team`  **Filter:**

```dax
[team_lead_email] = USERPRINCIPALNAME()
```

**Table:** `fact_sqm_monthly` — *no filter needed.* It joins `dim_team` directly,
so the scorecard is restricted by the same rule.

Filter flows `dim_team → dim_agent → fact_interactions`, which is exactly why
`dim_team` exists as a conformed dimension rather than a text column on
`dim_agent` (see `model_relationships.md`). One filter secures both facts.

### If leads should also see their own contacts as an agent

The rule above is keyed on *leading* a team. A lead who also takes contacts is
covered automatically — their `dim_agent` row carries the team's `team_key`.

### If a lead runs more than one team

The filter already handles it: `=` on a column returns every matching row, so a
lead listed against two teams sees both.

---

## Setting the roles up in the Power BI service

You can create roles in the browser — Power BI Desktop is not required. Dynamic
rules using `USERPRINCIPALNAME()` **must** be written in the DAX editor; the
default drop-down editor cannot express them.

1. Workspace → the semantic model → **⋯ → Open data model**.
2. Ribbon → **Manage roles**.
3. **New role** → name it exactly `CCR`.
4. Select table `dim_agent` → switch to the **DAX editor** → paste
   `[agent_email] = USERPRINCIPALNAME()` → **Save**.
5. **New role** → name it `Team Lead` → table `dim_team` → DAX editor →
   `[team_lead_email] = USERPRINCIPALNAME()` → **Save**.
6. Assign people: workspace → semantic model → **⋯ → Security** → pick the role
   → add users or a security group → **Add** → **Save**.

### Testing

In the data model editor, **Manage roles → Test as role**. Choose the role, and
optionally "Other user" with a specific email, to see the report as they would.

Expected results with this synthetic data:

| Test | Expect |
|---|---|
| `CCR` as `robert.johnson@prairieskyutilities.example.com` | ~1 agent, roughly 100 contacts |
| `Team Lead` as `danielle.johnson@prairieskyutilities.example.com` | 5 agents, Billing Care only |
| `Team Lead` — any other team's page | blank visuals, not an error |
| No role | all 20 agents, 2,000 contacts |

---

## Three things that will catch you out

**1. RLS does not apply to you if you own the workspace.** Workspace Admins,
Members and Contributors bypass RLS entirely — it only constrains **Viewers**.
Testing by opening the report yourself proves nothing. Use *Test as role*, or a
genuine Viewer account.

**2. `USERPRINCIPALNAME()` returns the signed-in UPN, which is not always the
email.** They differ in plenty of tenants. If filtering returns nothing, drop a
card with `USERPRINCIPALNAME()` on a test page and compare it against
`dim_agent[agent_email]` — a mismatch there is the usual cause. Build the column
from the UPN, not from a `firstname.lastname` convention, in anything real.

**3. A user in no role sees nothing; a user in two roles sees the union.** Roles
are additive, not intersecting. Someone in both `CCR` and `Team Lead` gets the
whole team, not just themselves.

---

## Verifying it actually works

Security that has not been tested from the outside is not security. Before
sharing the report:

- [ ] Open as a real **Viewer** account, not just *Test as role*.
- [ ] Confirm a CCR cannot see another agent anywhere — check the Team & Coaching
      page and the drill-through, not only the scorecard.
- [ ] Confirm the drill-through page respects RLS. It does inherit the filter
      context, but it is the page most likely to expose a single row of someone
      else's data.
- [ ] Decide explicitly whether team-level FCR is sensitive, and if it is, add the
      `dim_team` filter to the CCR role.
- [ ] Check the coaching note column. It is free text about an individual's
      performance, and the most sensitive field in the model.
