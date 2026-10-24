export interface ImageSize { width: number; height: number }
export interface NativeMaskRLE {
  encoding: 'BINARY_RLE_C_ORDER'
  coordinate_system: 'NATIVE_RESULT_MASK_GRID'
  size: [number, number] // native height, width, NOT original image dimensions
  counts: number[]
  starts_with: 0
}
export interface ModelIdentity { file: string | null; sha256: string | null; task: string | null; input_size: number | null }
export interface ResponseV2Fields {
  schema_version: '2.0.0'
  original_image_size: ImageSize
  polygon_coordinate_system: 'ORIGINAL_IMAGE_XY_PIXELS'
  polygon_representation: string
  full_image_classification: { source: string; is_primary: boolean; status: string; raw_top1_confidence: number | null; ood_rejection_status: 'NOT_VALIDATED' }
  seven_class_scores: { index: number; name: string; score: number }[] | null
  top1_index: number | null
  top1_name: string | null
  classification_threshold: number
  confidence_gate_status: 'ABOVE_THRESHOLD' | 'BELOW_THRESHOLD' | 'UNAVAILABLE' | 'INVALID_RESULT'
  segmentation_model_identity: ModelIdentity
  classification_model_identity: ModelIdentity
  segmentation_instance_count: number
  segmentation_instances: { index: number; mask: NativeMaskRLE | null; polygon: [number, number][] | null; predicted_box_index: number | null }[]
  predicted_bounding_boxes: { index: number; xyxy: [number, number, number, number] | null; confidence: number | null; class_index: number | null; coordinate_system: string }[]
  segmentation_status: string
  merged_roi: { x1: number; y1: number; x2: number; y2: number } | null
  merged_roi_kind: string
  review_required: true
  care_mapping_status: 'NOT_CONFIGURED'
  measurement_status: 'NOT_VALIDATED_NO_CALIBRATED_SCALE'
}
export interface PredictionDisplay extends Partial<ResponseV2Fields> {
  image: string
  classes: string[]
  class_confidence?: number | null
  requires_human_review: boolean
}
