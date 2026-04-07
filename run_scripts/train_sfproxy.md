# `train_sfproxy.sh` Sampler Manual

This note is a practical guide for choosing the sampler-related settings at the
top of `run_scripts/train_sfproxy.sh`.

It focuses on the data sampler, because for SFProxy this part is not a
small detail. The sampler decides what note events the model sees during
training: pitch coverage, chord density, onset spacing, duration distribution,
and especially velocity structure. In practice, many "model" differences are
actually data-generation differences.

## 1. What the sampler controls

When `train_sfproxy.sh` runs, it does three things:

1. Export synthetic data from a SoundFont.
2. Train one SFProxy model on that exported data.
3. Save checkpoints and log the run to W&B.

The sampler belongs to step 1. It decides how note events are sampled before
rendering with the SoundFont.

The main sampler-related knobs in `train_sfproxy.sh` are:

- `INSTRUMENT`
- `SEGMENT_SECONDS`
- `SAMPLER_PRESET`
- `BOUNDARY_MODE`
- `DISCOVER_BOUNDARIES`
- `SOUNDFONT` or `INSTRUMENT_PATH`
- `BOUNDARY_JSON`

If you want to think like a researcher, a good mental model is:

- `SAMPLER_PRESET` decides the overall distribution family
- `BOUNDARY_MODE` decides how velocity boundaries are handled
- `SEGMENT_SECONDS` decides the temporal task difficulty
- `SOUNDFONT` or `INSTRUMENT_PATH` decides which actual instrument response the model must learn

## 2. The sampler families

Under the hood there are three sampler types:

- `coverage`
  - hand-designed coverage sampler
  - directly samples pitch, duration, IOI, chord size, and velocity
  - good for broad control and explicit stress testing

- `realism`
  - corpus-statistics sampler
  - pitch, duration, IOI, chord-size distributions come from a real MIDI stats JSON
  - better when you want note-event statistics to resemble a real dataset

- `mixed`
  - mixture of multiple samplers with weights
  - useful when one sampler is too narrow and one sampler is too biased

In actual use, you usually choose a preset rather than the raw type.

## 3. Preset-by-preset explanation

### `coverage_shared_legacy`

This is the old baseline.

What it does:

- broad pitch / duration / IOI coverage
- hand-designed event distribution
- chord velocity mode is `shared`

What `shared` means:

- all notes in the same chord onset get the same velocity

Why it exists:

- baseline compatibility
- useful when you want to compare against the older Route IV data-generation idea

Strengths:

- simple
- reproducible
- good as a historical baseline

Weaknesses:

- chord velocity structure is too simplistic
- weaker for learning note-wise dynamics inside chords
- usually not the strongest choice for a final model

Use it when:

- you want a legacy baseline
- you want a controlled "old recipe" comparison

Avoid using it when:

- your main goal is best final SFProxy quality
- you care about realistic within-chord velocity variation

### `realism_shared_legacy`

This is the realism counterpart of the old baseline.

What it does:

- draws pitch / duration / IOI / chord size from a real corpus stats JSON
- still keeps chord velocity mode as `shared`

Strengths:

- more corpus-shaped than `coverage_shared_legacy`

Weaknesses:

- still inherits the old shared-chord velocity assumption
- usually not a good default mainline recipe

Use it when:

- you specifically want to isolate "realism stats" from "velocity-structure upgrade"

Avoid using it when:

- you want the new default SFProxy training recipe

### `coverage_v2`

This is the upgraded coverage sampler.

What changed relative to legacy:

- chord velocity mode defaults to `mixed`
- can use SoundFont-aware velocity boundaries
- can use register-aware boundaries rather than one global split

Strengths:

- keeps strong coverage
- much better for studying velocity modeling than legacy
- very useful for controlled ablations

Weaknesses:

- still hand-crafted rather than corpus-shaped
- can be less "natural" than a realism-based sampler

Use it when:

- you want a clean controlled experiment
- you want to test `fixed` vs `discovered` boundaries
- you want a strong single-family coverage baseline

This is probably the cleanest sampler for boundary ablations.

### `realism_v2`

This is the upgraded realism sampler.

What it does:

- pitch / duration / IOI / chord sizes come from real dataset statistics
- chord velocity mode defaults to `mixed`
- supports register-aware or discovered boundaries

Strengths:

- more realistic note-event distribution
- better than legacy realism for note-wise velocity structure

Weaknesses:

- if the underlying stats JSON is biased, your sampler inherits that bias
- narrower coverage than an intentionally broad coverage sampler

Use it when:

- you want training data closer to a real corpus
- you want to study whether realism helps generalization

Be careful when:

- your real stats JSON is from a dataset that mismatches your target task

### `mixed_v2`

This is the default recommended preset.

It is a mixture of four components:

- `boundary`
  - low-polyphony, boundary-focused coverage
  - intentionally oversamples around velocity boundaries

- `coverage`
  - broad hand-crafted coverage

- `realism`
  - corpus-driven event statistics

- `stress`
  - denser overlap, harder chords, more extreme settings

Why this is usually the best default:

- no single sampler is asked to do everything
- it balances coverage, realism, and hard cases
- it trains the proxy on both common and failure-prone regions

Strengths:

- strongest general-purpose default
- usually the best place to start for a serious run

Weaknesses:

- less clean as a controlled ablation
- harder to interpret than a single-family sampler

Use it when:

- you want your mainline model
- you are not trying to isolate one single data effect

Avoid using it when:

- you need the cleanest possible ablation story

## 4. Chord velocity modes

The v2 samplers can vary note velocities inside a chord in different ways.

### `shared`

- one velocity for the whole chord
- simplest
- legacy behavior

### `independent`

- each note gets its own sampled velocity
- strongest note-wise diversity

### `correlated`

- notes share a base velocity but each note gets jitter
- more realistic than pure shared, less chaotic than full independent

### `mixed`

- each chord randomly chooses one of `shared`, `independent`, or `correlated`
- this is the default in v2 presets

Practical interpretation:

- `shared` is good for old baselines
- `independent` is good for forcing note-wise discrimination
- `correlated` is a softer middle ground
- `mixed` is the most practical default

## 5. Velocity boundaries and `BOUNDARY_MODE`

Velocity boundaries are important because the SoundFont response is often not
uniform across the whole 0 to 127 velocity range. Some regions behave almost
the same, while some transition regions matter a lot.

The code supports two ideas:

- fixed default boundaries
  - legacy split points at roughly `0.33` and `0.66`

- discovered SoundFont-aware boundaries
  - estimated from the actual SoundFont response

### `BOUNDARY_MODE=default`

This means:

- use whatever is written in the preset config

For legacy presets this usually means:

- default fixed boundaries
- no discovered boundary JSON

For v2 presets this usually means:

- register-aware boundary strategy
- boundary JSON path defined in config

Important subtlety:

- if the boundary JSON path exists, `default` may behave very similarly to
  `discovered`
- if the JSON path is missing, the code falls back to the fixed boundaries

So `default` is convenient, but not always the cleanest ablation condition.

### `BOUNDARY_MODE=fixed`

This forces:

- no boundary JSON path
- boundary strategy becomes `global`

This is the clean baseline for:

- "do we really need SoundFont-aware boundaries?"

Use it when:

- you want a strict non-discovered baseline
- you are comparing directly against `discovered`

### `BOUNDARY_MODE=discovered`

This uses:

- the specified `BOUNDARY_JSON`
- SoundFont-aware boundaries

If the JSON file is missing:

- and `DISCOVER_BOUNDARIES=1`, the script will generate it
- and `DISCOVER_BOUNDARIES=0`, the sampler falls back to default `[0.33, 0.66]`

This fallback is convenient for debugging, but dangerous for research if you
thought you were running a true discovered-boundary experiment.

My recommendation:

- for serious experiments, ensure the boundary JSON already exists
- or set `DISCOVER_BOUNDARIES=1` and watch the log carefully

## 6. `global` vs `register` boundaries

There are two boundary strategies underneath:

- `global`
  - one shared boundary set for all pitches

- `register`
  - different boundary sets for low / mid / high pitch regions

`fixed` ablations force `global`.

The v2 presets usually prefer `register`, because many SoundFonts respond
differently across pitch register. This is especially reasonable for piano.

Practical rule:

- if you want a clean classical baseline, use `fixed`
- if you want the best modern recipe, use `discovered`

## 7. How sampler pairs with segment length

`SEGMENT_SECONDS` is not just an engineering detail. It changes the event
density, temporal context, and memory pressure.

### `2s`

Best for:

- fastest debugging
- cleaner local note-dynamics learning
- most standard starting point

Recommended with:

- `mixed_v2`
- `coverage_v2`
- `realism_v2`

### `5s`

Best for:

- moderate context
- stronger overlap and temporal interaction

Recommended when:

- the 2-second model is stable and you want more context

### `10s`

Best for:

- hardest context
- more long-range overlap

Be careful:

- slower export and training
- heavier GPU memory load
- more confounded if you are also changing sampler family

Research advice:

- when you ablate segment length, keep sampler and boundary fixed
- when you ablate sampler, keep segment length fixed

## 8. How sampler pairs with SoundFont

The sampler is not independent from the SoundFont.

This matters because:

- the SoundFont defines the actual velocity response curve
- discovered boundaries are SoundFont-specific
- realism or coverage preferences may behave differently on piano and guitar

### Piano

Usually the most natural recipe is:

- `SAMPLER_PRESET=mixed_v2`
- `BOUNDARY_MODE=discovered`
- `SEGMENT_SECONDS=2` or `5`

For clean ablations:

- `coverage_v2 + fixed`
- `coverage_v2 + discovered`

### Guitar

Also benefits from v2 samplers, but usually has a smaller pitch range and
sparser polyphony assumptions than piano.

A good default is:

- `SAMPLER_PRESET=mixed_v2`
- `BOUNDARY_MODE=discovered`
- `SEGMENT_SECONDS=2`

If you want a cleaner analysis:

- start from `coverage_v2`

## 9. Recommended recipes

### Recipe A: Mainline default run

Use:

- `SAMPLER_PRESET=mixed_v2`
- `BOUNDARY_MODE=discovered`
- `SEGMENT_SECONDS=2`

Why:

- strongest balanced default
- easiest serious starting point

### Recipe B: Clean boundary ablation

Use two runs:

- `coverage_v2 + fixed`
- `coverage_v2 + discovered`

Why:

- same sampler family
- only boundary treatment changes

This is cleaner than comparing `coverage_shared_legacy` to `mixed_v2`.

### Recipe C: Legacy baseline comparison

Use:

- `coverage_shared_legacy + default`

Optional second run:

- `realism_shared_legacy + default`

Why:

- gives you historical baseline numbers

### Recipe D: Realism-focused run

Use:

- `realism_v2 + discovered`

Why:

- useful when you believe event statistics matter more than hand-crafted
  coverage

### Recipe E: Fast debugging run

Use:

- `SEGMENT_SECONDS=2`
- `TRAIN_SIZE` much smaller
- `VAL_SIZE` much smaller
- `SAMPLER_PRESET=coverage_v2`
- `BOUNDARY_MODE=fixed`

Why:

- less moving parts
- easier to sanity-check

## 10. Combinations to avoid

These are the main taboo combinations.

### Do not use `legacy + discovered`

Specifically:

- `coverage_shared_legacy + discovered`
- `realism_shared_legacy + discovered`

Reason:

- legacy presets do not implement the new SoundFont-aware boundary logic in the
  intended ablation sense
- `train_sfproxy.sh` explicitly blocks this combination

### Do not trust `discovered` if the JSON is missing and auto-discovery is off

If:

- `BOUNDARY_MODE=discovered`
- `BOUNDARY_JSON` does not exist
- `DISCOVER_BOUNDARIES=0`

then the code falls back to `[0.33, 0.66]`.

That run is no longer a real discovered-boundary experiment.

### Do not change all axes at once

For example, avoid changing together:

- `SAMPLER_PRESET`
- `BOUNDARY_MODE`
- `SEGMENT_SECONDS`
- `SOUNDFONT` or `INSTRUMENT_PATH`

If all of these move at once, you will not know what caused the result.

### Do not use `mixed_v2` when you need the cleanest causal story

`mixed_v2` is often the best model recipe, but not the best analysis recipe.

If your question is:

- "do discovered boundaries help?"

then use:

- `coverage_v2` or `realism_v2`

not:

- `mixed_v2`

### Do not assume realism stats are automatically good

`realism_v2` is only as good as the stats JSON driving it.

If the stats source mismatches your downstream task, the sampler can become
realistic in the wrong way.

## 11. A simple decision tree

If you want the best default:

- use `mixed_v2 + discovered`

If you want the cleanest boundary ablation:

- use `coverage_v2`, compare `fixed` vs `discovered`

If you want a historical baseline:

- use `coverage_shared_legacy + default`

If you want corpus-shaped event statistics:

- use `realism_v2 + discovered`

If you want the easiest debug run:

- use `coverage_v2 + fixed + 2s`

## 12. Suggested top-of-file settings in `train_sfproxy.sh`

### Stable default

```bash
DEFAULT_INSTRUMENT="piano"
DEFAULT_SEGMENT_SECONDS="2"
DEFAULT_SAMPLER_PRESET="mixed_v2"
DEFAULT_BOUNDARY_MODE="discovered"
DEFAULT_DISCOVER_BOUNDARIES="0"
```

### Clean boundary study

```bash
DEFAULT_INSTRUMENT="piano"
DEFAULT_SEGMENT_SECONDS="2"
DEFAULT_SAMPLER_PRESET="coverage_v2"
DEFAULT_BOUNDARY_MODE="fixed"
```

Then rerun with:

```bash
DEFAULT_BOUNDARY_MODE="discovered"
```

### Legacy baseline

```bash
DEFAULT_INSTRUMENT="piano"
DEFAULT_SEGMENT_SECONDS="2"
DEFAULT_SAMPLER_PRESET="coverage_shared_legacy"
DEFAULT_BOUNDARY_MODE="default"
```

## 13. Final practical advice

For actual research, I would recommend this order:

1. Start with `coverage_v2 + fixed + 2s` to sanity-check the whole pipeline.
2. Move to `coverage_v2 + discovered + 2s` for the clean boundary comparison.
3. Train `mixed_v2 + discovered + 2s` as the mainline model.
4. Only then explore `5s` or `10s`.

This order keeps debugging simple and preserves a clean ablation story.
