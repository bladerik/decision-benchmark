# decision-benchmark

Skill pre AI agenta (Claude Code, Codex a ďalšie, ktoré čítajú `SKILL.md`). Zistí, či sa na tvoju úlohu hodí **decision model** (Jev, GPT Luna Decisions, Perplexity Decider, Drex, Clef…), a ak áno, otestuje všetky naraz na tvojej vlastnej, ručne zaradenej vzorke.

Z videa CodingFry: **Šampión medzi AI, o ktorom si ešte nepočul**.

## Čo urobí

1. Spýta sa, na čo zvažuješ decision model, a povie, či sa hodí (rýchle rozhodnutie áno/nie, výber z možností), alebo či potrebuješ bežný model či agenta.
2. Pomôže ti pripraviť otázky a vzorku 50 až 100 položiek, pri ktorých už poznáš správnu odpoveď.
3. Pustí na ňu naraz všetky decision modely dostupné na OpenRouteri.
4. Dá ti tabuľku: presnosť, či má model pravdu, keď si je istý, rýchlosť a cenu za 1 000 položiek. K tomu odporúčanie, aký model zvoliť, od akej istoty mu veriť a kam poslať zvyšok.

## Inštalácia

Jedným príkazom pre Claude Code, Codex, Cursor, Gemini CLI, Cline a ďalších agentov:

```sh
npx skills add bladerik/decision-benchmark
```

Alebo to nechaj na agentovi. Napíš mu: „Nainštaluj si skill z https://github.com/bladerik/decision-benchmark“.

Potom v novom chate napíš: „Chcem otestovať decision modely na mojej úlohe.“

## Čo potrebuješ

- Python 3 (bez ďalších balíčkov).
- Účet na [OpenRouteri](https://openrouter.ai) s pár eurami kreditu. Samotný test väčšinou stojí pár centov.
- API kľúč v premennej prostredia `OPENROUTER_API_KEY`. Nevkladaj ho do chatu ani do súborov.

## Citlivé dáta

Vzorka sa posiela cez OpenRouter viacerým poskytovateľom modelov. Pri citlivých dátach (klienti, zmluvy, zdravie, financie) ju anonymizuj alebo test nerob. Skill sa ťa na to spýta.

## Licencia

MIT
