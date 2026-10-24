/** Display seam for existing response advice; raw API data remains untouched. */
export interface CareDisplayPayload {
  care_mapping_status?: string
  llm_advice?: string
  rag_guidance?: { id: number; title: string; recommendation: string }[]
}

export function ResearchCareDisplay({ prediction }: { prediction: CareDisplayPayload }) {
  // V1 has no approval status. Fail closed; an API label alone is not expert approval.
  const status = prediction.care_mapping_status ?? 'NOT_CONFIGURED'
  return <section className="v2-safety" data-testid="research-care-notice">
    <strong>目前尚未配置經審定的照護規則</strong>
    <span>AI 輸出僅供研究展示與人工覆核，不是正式傷口診斷或照護建議。</span>
    <span>照護規則狀態：{status === 'NOT_CONFIGURED' ? status : 'UNVERIFIED_NO_EXPERT_APPROVAL'}</span>
    <span>高 confidence 不等於診斷確認；UNKNOWN／OOD 拒判能力尚未驗證。</span>
    <span>缺乏可信比例尺與實體面積驗證，不提供可信傷口 cm²。</span>
  </section>
}
