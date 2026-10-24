import { useEffect, useRef, useState } from 'react'
import { mapNativeMaskToOriginal } from './maskMapping'
import type { PredictionDisplay } from './predictionV2'

const colors = [[13, 148, 136], [244, 114, 86], [139, 92, 246], [217, 153, 36]]

export function AnalysisState({ loading, error }: { loading: boolean; error: string }) {
  if (error) return <div className="v2-notice v2-error" role="alert">{error}</div>
  if (loading) return <div className="v2-notice" role="status">影像分析中…請等待；此時尚無新模型結果。</div>
  return null
}

export function PredictionV2Panel({ prediction, originalUrl }: { prediction: PredictionDisplay; originalUrl: string }) {
  const canvas = useRef<HTMLCanvasElement>(null)
  const [showMask, setShowMask] = useState(true)
  const [showBoxes, setShowBoxes] = useState(true)
  const [showPolygons, setShowPolygons] = useState(false)
  const [showROI, setShowROI] = useState(false)
  const [maskError, setMaskError] = useState('')
  const size = prediction.original_image_size
  const isV2 = prediction.schema_version === '2.0.0' && !!size
  const instances = prediction.segmentation_instances

  useEffect(() => {
    setMaskError('')
    const target = canvas.current
    if (!target || !size || !instances) return
    const context = target.getContext('2d')
    if (!context) { setMaskError('此瀏覽器無法繪製遮罩；未產生替代輪廓。'); return }
    context.clearRect(0, 0, size.width, size.height)
    if (!showMask) return
    try {
      const image = context.createImageData(size.width, size.height)
      instances.forEach((instance, index) => {
        if (!instance.mask) return
        const mapped = mapNativeMaskToOriginal(instance.mask, size)
        const color = colors[index % colors.length]
        mapped.forEach((bit, pixel) => {
          if (bit) image.data.set([...color, 120], pixel * 4)
        })
      })
      context.putImageData(image, 0, 0)
    } catch (error) {
      context.clearRect(0, 0, size.width, size.height)
      setMaskError(error instanceof Error ? error.message : '遮罩無法顯示；未生成替代輪廓。')
    }
  }, [size, instances, showMask])

  if (!isV2 || !size) return <section className="v2-panel" data-testid="v1-fallback" data-contract-version="V1">
    <div className="v2-heading"><div><p className="eyebrow">LEGACY RESPONSE</p><h3>V1 相容顯示</h3></div><span className="secure-tag">仍需人工覆核</span></div>
    <div className="v2-images"><figure><figcaption>原始上傳影像</figcaption><img src={originalUrl} alt="原始上傳影像" /></figure><figure><figcaption>V1 標註預覽（不是 mask payload）</figcaption><img src={prediction.image} alt="V1 標註預覽" /></figure></div>
    <p className="v2-notice">V1 回傳未提供七類完整分數、原生 mask 與實際分割實例數；不從合併 ROI 推造。</p>
    <p className="muted">研究性預測，不是正式診斷；照護規則未配置、實體面積未驗證。</p>
  </section>

  const count = prediction.segmentation_instance_count ?? 0
  const boxes = prediction.predicted_bounding_boxes ?? []
  const roi = prediction.merged_roi
  const rawConfidence = prediction.full_image_classification?.raw_top1_confidence
  const threshold = prediction.classification_threshold
  const supportedPolygons = prediction.polygon_coordinate_system === 'ORIGINAL_IMAGE_XY_PIXELS'
  return <section className="v2-panel" data-testid="prediction-v2" data-contract-version="2.0.0">
    <div className="v2-heading"><div><p className="eyebrow">RESEARCH OUTPUT · RESPONSE {prediction.schema_version}</p><h3>影像與模型原始輸出</h3></div><span className="secure-tag">原圖主要分類</span></div>
    <div className="v2-controls">
      <label><input type="checkbox" checked={showMask} onChange={e => setShowMask(e.target.checked)} />Mask Overlay</label>
      <label><input type="checkbox" checked={showBoxes} onChange={e => setShowBoxes(e.target.checked)} />Bounding Boxes</label>
      <label><input type="checkbox" checked={showPolygons} onChange={e => setShowPolygons(e.target.checked)} />Polygon 輪廓</label>
      <label><input type="checkbox" checked={showROI} onChange={e => setShowROI(e.target.checked)} />合併 ROI（非實例）</label>
    </div>
    <div className="v2-images">
      <figure><figcaption>原始上傳影像 · {size.width} × {size.height}</figcaption><img src={originalUrl} alt="原始上傳影像" /></figure>
      <figure><figcaption>Mask／框疊圖 · 研究性預測</figcaption><div className="v2-image-frame" style={{ aspectRatio: `${size.width} / ${size.height}` }}>
        <img src={originalUrl} alt="遮罩疊圖底圖" />
        <canvas ref={canvas} width={size.width} height={size.height} aria-label="原生 RLE 反映射遮罩" data-testid="mask-overlay" />
        <svg viewBox={`0 0 ${size.width} ${size.height}`} aria-label="原圖 polygon 與預測框" data-testid="geometry-overlay">
          {showBoxes && boxes.map(box => box.xyxy && <rect key={`box-${box.index}`} x={box.xyxy[0]} y={box.xyxy[1]} width={box.xyxy[2] - box.xyxy[0]} height={box.xyxy[3] - box.xyxy[1]} className="v2-predicted-box" />)}
          {showPolygons && supportedPolygons && instances?.map(instance => instance.polygon && <polygon key={`polygon-${instance.index}`} points={instance.polygon.map(point => point.join(',')).join(' ')} className="v2-polygon" />)}
          {showROI && roi && <rect x={roi.x1} y={roi.y1} width={roi.x2 - roi.x1} height={roi.y2 - roi.y1} className="v2-merged-roi" />}
        </svg>
      </div></figure>
    </div>
    <div className="v2-geometry-summary"><strong data-testid="instance-count">{count} 個模型分割實例</strong><span>{boxes.length} 個預測框</span><span>{roi ? '1 個合併顯示 ROI' : '無合併 ROI'}</span></div>
    {count === 0 && <p className="v2-notice" data-testid="empty-mask">沒有模型分割實例；不製造輪廓，也不代表影像一定沒有傷口。</p>}
    {count > 0 && !(instances?.some(instance => instance.mask)) && <p className="v2-notice">模型回報實例，但無可用 native mask 表示；未以 ROI 補成遮罩。</p>}
    {maskError && <p className="v2-notice v2-error" role="alert">{maskError}</p>}
    <p className="v2-footnote">Native grids：{instances?.map(instance => instance.mask ? `${instance.mask.size[1]}×${instance.mask.size[0]}` : 'null').join('、') || '無'}。依現行置中 letterbox 反映射，僅作 nearest-neighbour 視覺化；RLE 保持原樣。Polygon 可能缺洞或連接斷裂區域，不是 mask 的無損替代。</p>
    <div className="v2-result-grid"><div>
      <div className="v2-top1"><span>Top-1 · 原圖封閉集合分數</span><strong>{prediction.top1_name ?? 'UNAVAILABLE'}</strong><span>{rawConfidence != null ? `${(rawConfidence * 100).toFixed(2)}%` : '無分數'} · Threshold {threshold?.toFixed(2) ?? 'UNAVAILABLE'}</span><b>{prediction.confidence_gate_status === 'ABOVE_THRESHOLD' ? '達到門檻，仍需覆核' : '低於門檻或結果不可用，需覆核'}</b></div>
      <table className="v2-scores" data-testid="seven-class-scores"><caption>七類完整 Confidence（不是診斷正確率）</caption><thead><tr><th>Index / Class</th><th>原始模型分數</th></tr></thead><tbody>{prediction.seven_class_scores?.map(score => <tr key={score.index}><td>{score.index} · {score.name}</td><td><progress value={score.score} max={1} aria-label={`${score.name} confidence`} /><span title={String(score.score)}>{(score.score * 100).toFixed(2)}%</span></td></tr>)}</tbody></table>
      {!prediction.seven_class_scores && <p className="v2-notice">完整分數不可用；未推造七類 scores。</p>}
    </div><div className="v2-models"><h4>模型身分與版本</h4>{[prediction.segmentation_model_identity, prediction.classification_model_identity].map((model, index) => <div key={index}><strong>{index === 0 ? 'Segmenter' : 'Classifier'} · {model?.file ?? 'UNAVAILABLE'}</strong><span>{model?.task ?? 'unknown'} · {model?.input_size ?? '?'} px</span><code>{model?.sha256 ?? '無 SHA256'}</code></div>)}<small>Response {prediction.schema_version} · OOD：{prediction.full_image_classification?.ood_rejection_status ?? 'NOT_VALIDATED'}</small></div></div>
    <div className="v2-safety" data-testid="medical-boundary"><strong>研究性預測，不是正式傷口診斷。人工覆核仍為必要。</strong><span>review_required = {String(prediction.review_required ?? prediction.requires_human_review)}</span><span>照護規則未配置 · {prediction.care_mapping_status ?? 'NOT_CONFIGURED'}</span><span>無可信比例尺，實體面積尚未驗證 · {prediction.measurement_status ?? 'NOT_VALIDATED_NO_CALIBRATED_SCALE'}</span><span>不計算 cm²；高 confidence 不代表診斷正確或已有可靠未知類拒判能力。</span></div>
  </section>
}
