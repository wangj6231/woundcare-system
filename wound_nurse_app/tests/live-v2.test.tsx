import { test } from 'node:test'
import assert from 'node:assert/strict'
import { decodeNativeMask, mapNativeMaskToOriginal } from '../src/maskMapping'
import { renderToStaticMarkup } from 'react-dom/server'
import { AnalysisState, PredictionV2Panel } from '../src/PredictionV2Panel'

test('native RLE is decoded losslessly and mapped to original image, not mislabeled as original raster', () => {
  const rle = { encoding: 'BINARY_RLE_C_ORDER' as const, coordinate_system: 'NATIVE_RESULT_MASK_GRID' as const, size: [2, 3] as [number, number], counts: [1, 2, 3], starts_with: 0 as const }
  assert.deepEqual(Array.from(decodeNativeMask(rle)), [0, 1, 1, 0, 0, 0])
  assert.deepEqual(Array.from(mapNativeMaskToOriginal(rle, { width: 6, height: 4 })), [0,0,1,1,1,1, 0,0,1,1,1,1, 0,0,0,0,0,0, 0,0,0,0,0,0])
})

test('centered-letterbox padding is removed rather than stretched into the source image', () => {
  const rle = { encoding: 'BINARY_RLE_C_ORDER' as const, coordinate_system: 'NATIVE_RESULT_MASK_GRID' as const, size: [8, 8] as [number, number], counts: [16,32,16], starts_with: 0 as const }
  assert.deepEqual(Array.from(mapNativeMaskToOriginal(rle, { width: 8, height: 4 })), Array(32).fill(1))
})

test('native holes remain empty and original RLE is never modified', () => {
  const rle = { encoding: 'BINARY_RLE_C_ORDER' as const, coordinate_system: 'NATIVE_RESULT_MASK_GRID' as const, size: [3, 3] as [number, number], counts: [0,4,1,4], starts_with: 0 as const }
  const before = JSON.stringify(rle)
  const mapped = mapNativeMaskToOriginal(rle, { width: 9, height: 9 })
  assert.equal(mapped[4*9+4], 0)
  assert.equal(mapped[0], 1)
  assert.equal(JSON.stringify(rle), before)
})

test('invalid RLE is rejected rather than generating a fake mask', () => {
  const rle = { encoding: 'BINARY_RLE_C_ORDER' as const, coordinate_system: 'NATIVE_RESULT_MASK_GRID' as const, size: [2, 3] as [number, number], counts: [10], starts_with: 0 as const }
  assert.throws(() => decodeNativeMask(rle), /RLE/)
})

test('empty mask is explicitly shown and never replaced by merged ROI', () => {
  const html = renderToStaticMarkup(<PredictionV2Panel originalUrl="blob:fixture-only" prediction={{ image:'fixture-only', classes:[], requires_human_review:true,
    schema_version:'2.0.0', original_image_size:{width:512,height:512}, segmentation_instance_count:0, segmentation_instances:[],
    predicted_bounding_boxes:[], merged_roi:{x1:10,y1:10,x2:90,y2:90}, seven_class_scores:null }} />)
  assert.match(html, /沒有模型分割實例/)
  assert.match(html, /0 個模型分割實例/)
  assert.doesNotMatch(html, /&lt;polygon/)
})

test('V1 original and annotated previews still render without invented scores', () => {
  const html = renderToStaticMarkup(<PredictionV2Panel originalUrl="blob:v1-original" prediction={{ image:'data:image/jpeg;base64,v1-fixture', classes:['Cut'], requires_human_review:true }} />)
  assert.match(html, /V1 相容顯示/)
  assert.match(html, /blob:v1-original/)
  assert.match(html, /v1-fixture/)
  assert.match(html, /V1 回傳未提供七類完整分數/)
  assert.doesNotMatch(html, /<progress/)
})

test('loading and escaped error states do not masquerade as prediction results', () => {
  assert.match(renderToStaticMarkup(<AnalysisState loading={true} error="" />), /分析中/)
  const html = renderToStaticMarkup(<AnalysisState loading={false} error="<bad image>" />)
  assert.match(html, /role="alert"/)
  assert.match(html, /&lt;bad image&gt;/)
  assert.doesNotMatch(html, /AI 分析完成/)
})

test('V2 panel shows true instance count, seven scores and unvalidated medical states', () => {
  const html = renderToStaticMarkup(<PredictionV2Panel originalUrl="blob:fixture-only" prediction={{
    image: 'data:image/jpeg;base64,fixture-only', classes: ['Abrasions'], requires_human_review: true,
    schema_version: '2.0.0', original_image_size: { width: 512, height: 512 },
    segmentation_instance_count: 2, segmentation_instances: [], predicted_bounding_boxes: [],
    top1_name: 'Abrasions', classification_threshold: .60, confidence_gate_status: 'BELOW_THRESHOLD',
    seven_class_scores: ['Abrasions','Bruises','Burns','Cut','Ingrown_nails','Laceration','Stab_wound'].map((name,index) => ({ name,index,score: index===0 ? .4 : .1 })),
    review_required: true, care_mapping_status: 'NOT_CONFIGURED', measurement_status: 'NOT_VALIDATED_NO_CALIBRATED_SCALE',
  }} />)
  assert.match(html, /2 個模型分割實例/)
  assert.match(html, /Stab_wound/)
  assert.match(html, /NOT_CONFIGURED/)
  assert.match(html, /NOT_VALIDATED_NO_CALIBRATED_SCALE/)
  assert.match(html, /0\.60/)
  assert.doesNotMatch(html, /已確認診斷/)
})
