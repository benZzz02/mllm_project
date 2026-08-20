# Generated Data Contract

Do not commit generated DGM4 records or images. Run the converter to create:

- `dgm4_sft_train.jsonl`: official train subset used by SFT.
- `dgm4_preference_pool.jsonl`: disjoint official train subset used for rollout and badcase mining.
- `dgm4_val.jsonl`: official validation split, evaluation only.
- `dgm4_test.jsonl`: official test split, final evaluation only.
- `dgm4_badcase_preference.jsonl`: SFT-model badcases for DPO/SimPO.
- `dgm4_base_badcase_preference.jsonl`: Base-model badcases for direct-DPO ablation.
- `dataset_info.json`: LLaMA-Factory dataset registrations.
- `manifest.json`: split parameters, counts and source paths.

SFT rows use multimodal ShareGPT:

```json
{
  "id": "stable-id",
  "images": ["/absolute/path/image.jpg"],
  "conversations": [
    {"from": "human", "value": "<image>\nNews text: ..."},
    {"from": "gpt", "value": "{\"verdict\":\"manipulated\",...}"}
  ],
  "meta": {"dgm4_split": "train", "pool": "sft_train", "target": {}}
}
```

Preference rows retain the human turn and add ShareGPT `chosen`/`rejected` assistant turns. The builder rejects any source row not marked `dgm4_split=train` and `pool=preference_pool`.

