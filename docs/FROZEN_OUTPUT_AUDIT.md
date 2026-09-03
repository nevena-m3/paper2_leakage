# Frozen output audit

This record summarizes the release-level checksum and seal audit performed against the archived `outputs.zip` supplied with the frozen analysis.

## Archive identity

- archive: `outputs.zip`
- bytes: `176301940`
- SHA-256: `adc2dd5661c182bd7acec20b17ddcf5d0a90de5f495fb1f6e9721428a16ffd4e`

## Git scientific freeze

- tag: `paper2-analysis-freeze-v1.0.0`
- commit: `55cba2083327aa71be5ea7771a3185daefa0c979`

The tag is retained as an immutable historical scientific freeze. The post-freeze provenance patch does not move this tag.

## Goal 1

- `DONE.json` SHA-256: `be91b3c8c54846d509bce80d9fec56faeeca1bb88ede65acfc39a171f313e337`
- `GOAL1_FINAL_FREEZE.json` SHA-256: `51dbbf354a9c45b5c37e21e6dba66be6e7a919a580b240e38f6d74f4dc941d55`
- completion manifest SHA-256: `a32d23989ab2ce0f2aa889a59c99c9bb666b4997c471a928bf2e0528ace3387c`
- all 37 artifact paths recorded by the Goal-1 final freeze were present and matched their recorded SHA-256 values.
- the saved OOF predictions independently reproduce the frozen diagnosis and severity metrics.
- the saved split table satisfies participant-disjoint outer-test assignment and complete repeated-fold coverage.
- the final diagnosis and severity permutation tables contain 1,000 null replicates per task and reproduce empirical `p = 1/1001 = 0.000999000999...`.

## Goal 2

- `DONE.json` SHA-256: `15460b9166b50e50a86d09ee164c4342388ad524f9e34b527d74f8ea67088a19`
- `GOAL2_COMPLETION_MANIFEST.json` SHA-256: `8cbd1f50d83f0fd50284c029d11d748dfaa7efe01db4ad584c39e47cfe41ea68`
- all 37 artifact paths recorded by the final Goal-2 package were present and matched their recorded SHA-256 values.
- saved authoritative OOF predictions independently reproduce the final diagnosis and severity metrics to floating-point precision.
- the paired participant bootstrap package contains 2,000 replicates for each reported task/contrast/metric and reproduces the saved confidence intervals.
- the authoritative completion outputs use the corrected fold-local residualizer tuning rule; the earlier fixed-alpha implementation remains historical only.

## Goal 3

- final `DONE.json` status: `DONE`
- `GOAL3_FINAL_FREEZE.json` SHA-256: `c0b7aaa4b4531bfcb118decdff7fd8e4247075f01562a0fba4b8923b0643c29e`
- final figure manifest SHA-256: `1b93fcf3c831109824143daa8c8c5be4d248f383f999108a24e07989dbe790a4`
- Stage-E controlled response SHA-256: `9b24dfea5115d2f852f3098c86bb2e94b410724ad29d4b8253589e2f41acf584`
- Stage-E bootstrap table SHA-256: `01547049019c25eecc71bc59b094072bc139d991f68925867c373ef0ba2fe40d`
- target-Q order audit SHA-256: `9facdce4cda50e4da418cc965a30f6c1a855922bcc09b97db298f7b0e21636a2`
- HGB Goal-2 reproduction audit SHA-256: `8cb29df38cd133137e88f41c8b95b60cfc2412fcfd16f43aa3026391a323f985`
- full controlled response: 14,792/14,792 expected rows; zero measurement-unavailable rows.
- target-Q low→medium→high ordering: PASS for all task/transform branches.
- target-Q expected-direction audit: PASS for all task/transform branches.
- participant bootstrap package: 2,000 replicates.
- frozen HGB Goal-2 reproduction: PASS; maximum absolute prediction discrepancy `1.7763568394002505e-15`.
- perturbed prediction unavailability: `M_A = 0`, `M_A+Q = 191`, `M_A-resQ = 191`; baseline prediction unavailability is zero. The Q-dependent failures were retained rather than imputed or repaired by model refitting.
- all 12 final figure assets (4 figures × PNG/PDF/SVG) and the approved-version source tables matched their frozen manifest hashes.

## Scope of this audit

This is a frozen-artifact/code/provenance audit, not a public raw-data clean-room reproduction. Restricted source recordings and participant-level data are not distributed in this repository. Therefore, an independent third party without governed source access cannot regenerate the project from raw audio solely from the public GitHub repository.
