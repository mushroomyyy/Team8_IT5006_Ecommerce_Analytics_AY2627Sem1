# Running TeamFuse for this project

This folder is a vendored copy of [agentdmai/teamfuse](https://github.com/agentdmai/teamfuse)
(MIT) with the company brief (`agents/sop/company.md`) and the Analyst agent
prefilled for the Olist late-delivery project. All five roles (PM, Eng, QA,
Marketing, Analyst) are kept, and the Analyst doubles as the "ask anything"
bot (see "Ask-anything mode" in `agents/analyst/CLAUDE.md`).

TeamFuse runs on your own machine: it needs an interactive AgentDM OAuth login
and long-lived local `claude` processes, so it cannot be started from a cloud
session.

## Steps

```bash
cd teamfuse
cp .mcp.json.example .mcp.json
cd agents-web && npm install && cd ..
claude
> /plugin install agentdm@agentdm
> /reload-plugins
> /mcp        # AgentDM -> Authenticate, connect as "Admin Agent"
> /teamfuse-init
```

At the `/teamfuse-init` prompts:

* Company name / brief: keep the existing `agents/sop/company.md` (it is
  already filled in).
* Roles: all five.
* GitHub org: `mushroomyyy`; wire this repo's local clone into `@eng-bot` and
  `@qa-bot` (absolute path to the parent repo).
* Postgres DSN: skip. There is no database; the Analyst reads the bundled
  Olist data.

Then `cd agents-web && cp .env.example .env.local && npm run dev` and open
http://127.0.0.1:3005.

## Known gap

The upstream README and `agents-web/src/lib/supervisor.ts` reference
`scripts/agent-loop.py` (the streaming wrapper), but that file is not present
in the upstream repo at the time of vendoring. Starting agents from the panel
will fail until it is added; check upstream for it or ask the maintainers.
