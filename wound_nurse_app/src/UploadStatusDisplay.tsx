import { AnalysisState } from './PredictionV2Panel'
import type { UploadSnapshot } from './uploadSession'

export function UploadStatusDisplay<T>({ snapshot }: { snapshot: UploadSnapshot<T> }) {
  return <div data-testid="upload-status" data-upload-status={snapshot.status}>
    {snapshot.status==='validating' && <div className="v2-notice" role="status">正在檢查新影像…舊結果已清除。</div>}
    <AnalysisState loading={snapshot.status==='loading'} error={snapshot.status==='error' ? snapshot.error ?? '影像分析失敗' : ''} />
    {snapshot.input && <p className="v2-footnote" data-testid="input-identity">目前輸入：{snapshot.input.name} · {snapshot.input.size} bytes · 本機輸入 {snapshot.input.id} · 請求 {snapshot.requestId ?? '尚未送出'} · {snapshot.status==='success' ? '結果僅屬於此輸入與請求' : '尚無本次成功結果'}。識別碼為本機序號，並非影像雜湊或伺服器 request ID。</p>}
  </div>
}
