# CHANGELOG.md
All notable changes to this project will be documented in this file.

## [Unreleased] → `v0.13.4` 
### Planned
- **Clean-up of bugs and re-synthesis with thorough testing**  
> *Release branch:* `v0.13.4`  
> *ETA:* **2025-07-25**

----  

### Added
- **3-D keyword network visualiser** [brief description](https://github.com/oldwalls/sapphire/blob/main/MAP_semantic_UMB_map.md) – top-20 nodes with force-directed layout in 3-space.

### Fixed
- Replaced `…` delimiter with `. ` in memory concatenation
  to restore CSCSR lexical overlap and stabilise identity loops.

### Improved
- CSCSR retrieval: removed extra `\n` token between prompt and inference,
  yielding ~12 % higher cosine scores in quick bench.

## [0.13.3] – 2025-07-17

### Added
- Programmable `prompt_constr` template.

### Changed
- Soft-logit bias scale `lam` default set to `0.65`.

---

### Research
- **Recursive sieve toggle** (`config set recursive_sieve 1`)
  Allows the sampler to keep resampling until cosine ↑ and utterance length ↓ (early closure test).
