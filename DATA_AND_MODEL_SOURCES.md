# Sources, attribution, and access

Read the upstream terms before downloading or using each source. The Apache-2.0
code license does not apply to third-party data, weights, or the IEEE template.
No access token, model weight, ChatQA source corpus, or transformed IHEval input
is committed. The preparation scripts download the original files by immutable
revision and verify their SHA-256 values before local reconstruction.

| Source | Revision | Role in this release |
|---|---|---|
| [ChatQA Training Data](https://huggingface.co/datasets/nvidia/ChatQA-Training-Data) | `164f018cf5e7862ae5ef9c2b8ada47b07e3e3a31` | Synthetic conversational QA training; upstream specifies non-commercial use and additional conditions in its source card. |
| [IHEval](https://github.com/ytyz1307zzh/IHEval) | `726a62924c3050045954df94347d53fe2bd1090d` | Clean utility inputs and native scorers; upstream license is CC BY-NC-ND 4.0. Original sources and locally reconstructed cases remain outside Git. |
| [MTRAG](https://github.com/IBM/mt-rag-benchmark) | `53fef6f5917c7e10c436f697e69a955dbc795d81` | Fixed reference history; repository includes Apache-2.0 terms and source attribution. |
| [BIPIA](https://github.com/microsoft/BIPIA) | `a004b69ec0dd446e0afd461d98cb5e96e120a5d0` | Clean email/table QA; MIT license. |
| [ASIDE](https://github.com/egozverev/aside) | `dfa94270d48b006decdcd4d455df3e10d6004880` | SEP source cases; Apache-2.0 repository. |

The full file-level source lock is `reproducibility/sources.lock.json`.
Source-row indices, decisions, and hashes reconstruct the same training split
without committing the 274 MB ChatQA source or its prepared conversations.
The generated-response archives contain model output token IDs, not source
prompts or reference answers. Retain upstream attribution and applicable source
conditions when using or redistributing resulting research artifacts.

Models are fetched from Hugging Face under their respective model-card terms:

| Model | Immutable revision |
|---|---|
| `Qwen/Qwen2.5-7B` | `d149729398750b98c0af14eb82c78cfe92750796` |
| `Qwen/Qwen2.5-7B-Instruct` tokenizer | `a09a35458c702b33eeacc393d103063234e8bc28` |
| `ISTA-MLCV/Qwen2.5-7B_single_emb` | `f73f5bf71de38503853b81ff8993cffebb974cd3` |
| `ISTA-MLCV/Qwen2.5-7B_aside` | `eaaaa196b9159b9e5ca686994e1a97110d988913` |
| `ISTA-MLCV/Qwen2.5-7B_ise` | `6e9f2fe749572ed2200de4490fc62be1ef55ec8b` |

Authentication, if a provider requires it, belongs in the local environment or
credential store. Never place credentials in a config file or commit them.
