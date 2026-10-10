# Licence notes for the NKI 0.7.0b1 Reference Skill

This directory combines material under three licences. This file records what came from where and what each licence requires. It is not legal advice. The study owner decides on publication.

## 1. aws-neuron/neuron-agentic-development: Apache License 2.0

- Source: https://github.com/aws-neuron/neuron-agentic-development, commit `ee25e45b9ace4aaefd8b8913bf5fa9b68adf6674`.
- Files here: `LICENSE.apache-2.0` (byte-identical copy of upstream `LICENSE.txt`) and `NOTICE.neuron-agentic-development` (byte-identical copy of upstream `NOTICE`: "Neuron Agentic Development / Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.").
- Use: rules and lists of stale or renamed APIs, and the fori/while-loop rules, were paraphrased after re-checking each against the installed package and the device. No code was copied.
- Obligations met: a copy of the licence and the NOTICE are distributed with the derivative text, and `SKILL.md` starts with an attribution that states the text was modified and condensed (Apache-2.0 §4(a), (b), (d)).

## 2. Installed `nki` 0.7.0b1 package: Apache License 2.0

- The wheel metadata (`nki-0.7.0b1+34883233359.g2aa6a82a.dist-info/METADATA`) declares `License-Expression: Apache-2.0`. Its `licenses/NOTICE` reads "Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved." Its `licenses/LICENSE` is the Apache-2.0 text (SHA-256 `09e8a9bcec8067104652c168685ab0931e7868f9c8284b66f5ae6edae5f1130b`; same licence as `LICENSE.apache-2.0` here).
- Use: API names, signatures, defaults, short docstring sentences and validation messages are quoted in `SKILL.md`, `COMPATIBILITY.md` and `evidence/introspection.json` (the full docstrings are in the JSON).
- Note for the study owner: the wheel's `Project-URL` points to `https://github.com/aws-neuron/private-nki-staging`, a private repository, and this 0.7.0b1 build is a pre-release. On 2026-10-10 the public Neuron pip index (https://pip.repos.neuron.amazonaws.com/nki/) listed nki up to `0.6.0+31049202112.g85070674`; the PyPI project named `nki` is unrelated (versions 0.0.1 and 0.1). The licence in the wheel is Apache-2.0. Whether quoting a pre-release's API in a public artifact is acceptable is a policy decision outside this licence check.

## 3. AWS Neuron documentation: CC BY-SA 4.0

- Source: https://awsdocs-neuron.readthedocs-hosted.com (built from https://github.com/aws-neuron/aws-neuron-sdk). That repository's `LICENSE-DOCUMENTATION` is the Creative Commons Attribution-ShareAlike 4.0 International Public License. Its `LICENSE-SAMPLECODE` is MIT No Attribution. Both were checked on 2026-10-10 at master `e27a3986fa8625d4103a7ce611ee0c85ae79979a`.
- Use: hardware facts and descriptions were paraphrased and condensed in `SKILL.md` §2.3, §3.3, §5.1 and §8: SBUF/PSUM capacity, LNC organisation, the double-FP8 layout and RNE rounding. The NKI 0.6.0 release notes were also used.
- CC BY-SA 4.0 requirements for adapted material: (a) credit the source, (b) link the licence, (c) indicate changes, (d) license the adapted material under CC BY-SA 4.0 or a compatible licence.
  - (a), (b) and (c) are satisfied by the attribution block at the top of `SKILL.md` and by `SOURCES.md` §4.
  - (d): `SKILL.md` states that the parts adapted from AWS documentation are shared under CC BY-SA 4.0. Licence URL: https://creativecommons.org/licenses/by-sa/4.0/
- Hardware facts (numbers such as 28 MiB) are not themselves copyrightable. The ShareAlike notice is kept because the wording of some explanations follows the documentation.
- The neuron-agentic-development mirror of the docs (`skills/neuron-nki-docs/references/**`) is distributed under Apache-2.0 by that repository. Where both apply, attribution to both is given.

## 4. aws-neuron/nki-samples: MIT No Attribution (MIT-0)

- Source: https://github.com/aws-neuron/nki-samples at `62c300da468b34655aa1fc98a70d807c8dad4a72`. `LICENSE` is "MIT No Attribution", Copyright Amazon.com, Inc. or its affiliates.
- Use: only the licence, the NKI version the samples target (0.6.0), and the API style of the tensor-addition tutorial were checked. The element-wise-add example in `SKILL.md` §1.3 follows the same public tutorial pattern but was written and tested independently. MIT-0 requires no attribution. It is credited in `SOURCES.md` anyway.

## 5. Original material

The smoke tests, the introspection script and the code examples in `SKILL.md` were written for this skill and run on the study's trn2 host. They are covered by the TileBench repository's licence terms, unless the study owner releases them under one of the licences above.
