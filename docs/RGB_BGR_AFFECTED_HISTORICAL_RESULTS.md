# RGB/BGR input-contract audit — historical status (2026-09-20)

Ultralytics treats a PIL RGB input and a NumPy BGR input consistently through its loader/preprocessing; a NumPy array containing RGB bytes is **not** equivalent. Phase A used a distinct-channel synthetic pattern to establish this contract. Phase A.5 added `validate_model_input_contract` and corrected the ISIC → FUSeg gate to explicitly convert PIL RGB to BGR NumPy. No development cohort was reevaluated here.

| Script or route | Input route | Affected status | Historical result status | Future reevaluation? |
|---|---|---|---|---|
| `experiments/review_v2/isic_fuseg_gate.py` historical run | PIL RGB → NumPy RGB → `predict_materialized` | CONFIRMED_AFFECTED; code now corrected | Historical candidate-versus-baseline gate not a color-contract-controlled comparison | YES, controlled development reevaluation with frozen cohort and settings |
| `experiments/review_v2/localization_benchmark.py` GPU/CPU | PIL RGB → explicit RGB2BGR NumPy | NOT_AFFECTED_BY_THIS_BUG | Retain with other limitations | No for this bug |
| `experiments/review_v2/highres_inference_experiment.py` | PIL RGB → explicit RGB2BGR NumPy | NOT_AFFECTED_BY_THIS_BUG | Retain with other limitations | No for this bug |
| `experiments/review_v2/tiled_inference_experiment.py` | PIL RGB → explicit RGB2BGR NumPy | NOT_AFFECTED_BY_THIS_BUG | Retain with other limitations | No for this bug |
| `backend_main.py` existing inference route | `cv2.imdecode` BGR | NOT_AFFECTED_BY_THIS_BUG | Not an experimental performance claim | No for this bug |
| `experiments/scripts/evaluate.py` classification Phase 7 | image path string to model | NOT_AFFECTED_BY_THIS_BUG | Classification saved-array arithmetic separately corrected | No for this bug |
| `evaluate_dseg06_carch05_oof_cascade.py` | full-image/ROI RGB NumPy, including segmenter | CONFIRMED_AFFECTED | SUPERSEDED as color-contract-controlled evidence | YES before use as comparable cascade result |
| `evaluate_classifier_full_image_val.py`; `evaluate_segmentation_cascade_fallback.py` | classifier receives RGB NumPy | CONFIRMED_AFFECTED | SUPERSEDED as color-contract-controlled evidence | YES |
| `run_automatic_cascade_eval.py`; `run_clinical_bbox_cascade_inference.py`; `run_segmentation_cascade_eval.py` | classifier receives ROI RGB NumPy | CONFIRMED_AFFECTED | SUPERSEDED as color-contract-controlled evidence | YES |
| `run_cascade_threshold_sweep.py` | crop/full-image RGB NumPy | CONFIRMED_AFFECTED | SUPERSEDED for threshold selection under the intended input contract | YES, development-only controlled sweep; no locked/external retuning |
| Production cascade validation and clinical segmentation cascade routes noted in Phase A | `read_image` uses `cv2.imdecode` BGR | NOT_AFFECTED_BY_THIS_BUG | Other evidence limits remain | No for this bug |

This is an input-route audit, not a quantified causal estimate of metric change. Historical files are preserved. Do not reuse the locked classification test or CO2Wounds external cohort for retuning.
