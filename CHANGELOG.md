# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## v0.3.2 (2026-07-03)

### Fix

- use largest segment for multi segment recording (#42)

## v0.3.1 (2026-04-29)

### Perf

- remove unnecessary allocs

## v0.3.0 (2026-04-25)

### BREAKING CHANGE

- extract_continuous's metrics implementation switched
from a SpikeInterface filter chain to FFT-based computation in commit
ba9a482. Consumer-visible changes:

### Feat

- document FFT metrics rewrite as breaking change

## v0.2.2 (2026-03-24)

### Fix

- constrain parallelism in save_rms_and_lfp_spectrum

## v0.2.1 (2026-03-24)

### Fix

- breaking changes in spike interface 0.104 (#29)

## v0.2.0 (2026-02-26)

### Feat

- 25 adapt lfp correlation for multi-shank probes (#26)

### Fix

- extract probe names from streams (#15)

## v0.0.0 (2024-12-12)
