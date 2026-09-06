import { useEffect, useMemo, useRef, useState } from 'react'
import type { ChangeEvent, FormEvent, RefObject } from 'react'
import './App.css'

type Role = 'admin' | 'head_nurse' | 'nurse'
type Status = 'red' | 'yellow' | 'green'
type View = 'board' | 'assessment' | 'patients' | 'audit' | 'admin' | 'feedback'

interface Patient {
  id: number
  room: string
  name: string
  gender: string
  age: string
  admission_date: string
  medical_history: string
  allergies: string
  assigned_nurse?: string
  status: Status
  last_assessment: string
}

interface EMRRecord {
  id: number
  patient_id: number
  patient_name?: string
  nurse_name: string
  shift: string
  wound_image?: string | null
  ai_analysis?: string
  treatment: string
  status: Status
  review_status: 'reviewed' | 'needs_review'
  human_review_confirmed: boolean
  human_reviewed_at?: string | null
  timestamp: string
}

interface UserAccount {
  id: number
  username: string
  role: Role
  is_active?: number
  created_at?: string
}

interface Detection {
  box: { x1: number; y1: number; x2: number; y2: number }
  confidence: number
  label: string
  class_confidence: number | null
}

interface Prediction {
  image: string
  detected_boxes: number
  detections: Detection[]
  classes: string[]
  inference_mode?: 'segmentation_crop' | 'full_image_primary' | 'full_image_fallback' | 'legacy_detection'
  fallback_used?: boolean
  low_confidence?: boolean
  class_confidence?: number | null
  llm_advice: string
  rag_guidance: RAGGuidance[]
  requires_human_review: boolean
  model?: { file: string; sha256: string | null; input_size: number; inference_mode: string }
}

interface SystemHealth {
  status: string
  professor_preview: boolean
  detection_model_loaded: boolean
  segmentation_model_loaded: boolean
  classification_model_loaded: boolean
  inference_mode: string
  classification_model: { file: string; sha256: string | null }
  classification_input_size: number
}

interface ProfessorFeedbackItem {
  id: number
  username: string
  task: string
  usability_score: number
  clinical_clarity_score: number
  comments: string
  page: string
  created_at: string
}

interface RAGGuidance {
  id: number
  title: string
  wound_classes: string
  recommendation: string
  rationale: string
  precautions: string
  evidence_level: string
  source: string
  created_at: string
  updated_at: string
  match_score?: number
}

interface GuidanceDraft {
  title: string
  wound_classes: string
  recommendation: string
  rationale: string
  precautions: string
  evidence_level: string
  source: string
  source_emr_id: string
}

// In development the Vite proxy keeps the API same-origin, which also makes
// the page work when opened from a phone on the same LAN as the workstation.
const API_BASE = import.meta.env.VITE_API_BASE ?? ''
const roleLabel: Record<Role, string> = { admin: '系統管理員', head_nurse: '護理長', nurse: '臨床護理師' }
const statusLabel: Record<Status, string> = { red: '優先處理', yellow: '需追蹤', green: '穩定' }

function App() {
  const [token, setToken] = useState(() => sessionStorage.getItem('woundcare_token') ?? '')
  const [user, setUser] = useState<{ username: string; role: Role } | null>(() => {
    const saved = sessionStorage.getItem('woundcare_user')
    return saved ? JSON.parse(saved) : null
  })
  const [view, setView] = useState<View>('board')
  const [patients, setPatients] = useState<Patient[]>([])
  const [records, setRecords] = useState<EMRRecord[]>([])
  const [users, setUsers] = useState<UserAccount[]>([])
  const [systemHealth, setSystemHealth] = useState<SystemHealth | null>(null)
  const [audit, setAudit] = useState<Array<{ id: number; username?: string; action: string; resource?: string; created_at: string }>>([])
  const [feedback, setFeedback] = useState<ProfessorFeedbackItem[]>([])
  const [ragGuidance, setRagGuidance] = useState<RAGGuidance[]>([])
  const [selectedPatient, setSelectedPatient] = useState<Patient | null>(null)
  const [loading, setLoading] = useState(false)
  const [toast, setToast] = useState('')
  const [login, setLogin] = useState({ username: '', password: '' })
  const [loginError, setLoginError] = useState('')
  const [shift, setShift] = useState('白班')
  const [search, setSearch] = useState('')
  const [statusFilter, setStatusFilter] = useState<'all' | Status>('all')
  const [imagePreview, setImagePreview] = useState<string | null>(null)
  const [prediction, setPrediction] = useState<Prediction | null>(null)
  const [analysisFile, setAnalysisFile] = useState<File | null>(null)
  const [treatment, setTreatment] = useState('')
  const [status, setStatus] = useState<Status>('yellow')
  const [reviewStatus, setReviewStatus] = useState<'reviewed' | 'needs_review'>('needs_review')
  const [humanReviewConfirmed, setHumanReviewConfirmed] = useState(false)
  const [newPatient, setNewPatient] = useState({ room: '', name: '', gender: '', age: '', admission_date: new Date().toISOString().slice(0, 10), medical_history: '', allergies: '', assigned_nurse: '' })
  const [newUser, setNewUser] = useState({ username: '', password: '', role: 'nurse' as Role })
  const [guidanceDraft, setGuidanceDraft] = useState<GuidanceDraft>({ title: '', wound_classes: '', recommendation: '', rationale: '', precautions: '', evidence_level: 'clinical_consensus', source: 'head_nurse_review', source_emr_id: '' })
  const [feedbackDraft, setFeedbackDraft] = useState({ task: '整體教授測試', usability_score: 5, clinical_clarity_score: 5, comments: '' })
  const fileInput = useRef<HTMLInputElement>(null)

  const api = async (path: string, init: RequestInit = {}) => {
    const headers = new Headers(init.headers)
    if (token) headers.set('Authorization', `Bearer ${token}`)
    const response = await fetch(`${API_BASE}${path}`, { ...init, headers })
    if (response.status === 401) {
      handleLogout()
      throw new Error('登入工作階段已失效，請重新登入')
    }
    const data = await response.json().catch(() => ({}))
    if (!response.ok) throw new Error(data.detail ?? '操作失敗')
    return data
  }

  const showToast = (message: string) => { setToast(message); window.setTimeout(() => setToast(''), 3600) }

  const refresh = async () => {
    if (!token || !user) return
    setLoading(true)
    try {
      setSystemHealth(await api('/api/health'))
      const patientData = await api('/api/patients')
      setPatients(patientData)
      if (user.role !== 'nurse') {
        const allRecords = await api('/api/emr/all')
        setRecords(allRecords)
      } else if (selectedPatient) {
        setRecords(await api(`/api/patients/${selectedPatient.id}/emr`))
      }
      if (user.role === 'admin') setUsers(await api('/api/admin/users'))
      if (user.role !== 'nurse') { setAudit(await api('/api/audit')); setFeedback(await api('/api/professor-feedback')) }
      setRagGuidance(await api('/api/rag/guidance?limit=20'))
    } catch (error) { showToast(error instanceof Error ? error.message : '資料載入失敗') } finally { setLoading(false) }
  }

  // Refresh only when the authenticated principal changes; refresh itself is
  // intentionally kept local so it always reads the current session token.
  // eslint-disable-next-line react-hooks/exhaustive-deps
  useEffect(() => { void refresh() }, [token, user?.role])

  const filteredPatients = useMemo(() => patients.filter((patient) => {
    const query = search.trim().toLowerCase()
    const matchesSearch = !query || [patient.room, patient.name, patient.id.toString()].some((value) => value.toLowerCase().includes(query))
    return matchesSearch && (statusFilter === 'all' || patient.status === statusFilter)
  }), [patients, search, statusFilter])

  const counts = useMemo(() => ({ total: patients.length, red: patients.filter(p => p.status === 'red').length, yellow: patients.filter(p => p.status === 'yellow').length, green: patients.filter(p => p.status === 'green').length }), [patients])

  const handleLogin = async (event: FormEvent) => {
    event.preventDefault(); setLoginError('')
    try {
      const data = await fetch(`${API_BASE}/api/login`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(login) }).then(async response => { const body = await response.json(); if (!response.ok) throw new Error(body.detail ?? '登入失敗'); return body })
      const nextUser = { username: data.username, role: data.role as Role }
      sessionStorage.setItem('woundcare_token', data.token); sessionStorage.setItem('woundcare_user', JSON.stringify(nextUser))
      setToken(data.token); setUser(nextUser); setLogin({ username: '', password: '' })
    } catch (error) { setLoginError(error instanceof Error ? error.message : '登入失敗') }
  }

  const handleLogout = () => {
    if (token) void fetch(`${API_BASE}/api/logout`, { method: 'POST', headers: { Authorization: `Bearer ${token}` } }).catch(() => undefined)
    sessionStorage.removeItem('woundcare_token'); sessionStorage.removeItem('woundcare_user'); setToken(''); setUser(null); setSelectedPatient(null); setPrediction(null)
  }

  const handleImage = (event: ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0]
    if (!file) return
    if (!['image/jpeg', 'image/png', 'image/webp'].includes(file.type) || file.size > 10 * 1024 * 1024) { showToast('請選擇 10 MB 以內的 JPG、PNG 或 WebP 圖片'); return }
    setAnalysisFile(file); setPrediction(null); setHumanReviewConfirmed(false); setReviewStatus('needs_review'); setImagePreview(URL.createObjectURL(file))
  }

  const runAnalysis = async () => {
    if (!analysisFile) return
    setLoading(true)
    try { const form = new FormData(); form.append('file', analysisFile); const result = await api('/api/predict', { method: 'POST', body: form }) as Prediction; setPrediction(result); setHumanReviewConfirmed(false); if (result.low_confidence || result.fallback_used) setReviewStatus('needs_review'); showToast(result.low_confidence || result.fallback_used ? '此結果需要護理長或臨床人員進一步覆核' : '分析完成，請由護理人員確認後存檔') } catch (error) { showToast(error instanceof Error ? error.message : '影像分析失敗') } finally { setLoading(false) }
  }

  const saveAssessment = async () => {
    if (!selectedPatient || !treatment.trim()) { showToast('請選擇病人並完成處置紀錄'); return }
    if (reviewStatus === 'reviewed' && !humanReviewConfirmed) { showToast('請先明確確認已完成人工覆核'); return }
    try {
      await api(`/api/patients/${selectedPatient.id}/emr`, { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ shift, wound_image: prediction?.image ?? null, ai_analysis: prediction?.llm_advice ?? null, treatment, status, review_status: reviewStatus, human_review_confirmed: humanReviewConfirmed }) })
      showToast('評估已安全儲存'); setTreatment(''); setPrediction(null); setImagePreview(null); setHumanReviewConfirmed(false); setReviewStatus('needs_review'); setView('board'); await refresh()
    } catch (error) { showToast(error instanceof Error ? error.message : '儲存失敗') }
  }

  const createPatient = async () => {
    try { await api('/api/patients', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(newPatient) }); setNewPatient({ room: '', name: '', gender: '', age: '', admission_date: new Date().toISOString().slice(0, 10), medical_history: '', allergies: '', assigned_nurse: '' }); showToast('病人資料已建立'); await refresh() } catch (error) { showToast(error instanceof Error ? error.message : '建立失敗') }
  }

  const createUser = async () => {
    try { await api('/api/admin/users', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify(newUser) }); setNewUser({ username: '', password: '', role: 'nurse' }); showToast('帳號已建立'); await refresh() } catch (error) { showToast(error instanceof Error ? error.message : '建立失敗') }
  }

  const saveGuidance = async () => {
    if (!guidanceDraft.title.trim() || !guidanceDraft.recommendation.trim()) { showToast('請填寫建議標題與處置內容'); return }
    try {
      await api('/api/rag/guidance', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...guidanceDraft, source_emr_id: guidanceDraft.source_emr_id ? Number(guidanceDraft.source_emr_id) : null }) })
      setGuidanceDraft({ title: '', wound_classes: '', recommendation: '', rationale: '', precautions: '', evidence_level: 'clinical_consensus', source: 'head_nurse_review', source_emr_id: '' })
      showToast('護理長建議已寫入 RAG 知識庫'); setRagGuidance(await api('/api/rag/guidance?limit=20'))
    } catch (error) { showToast(error instanceof Error ? error.message : 'RAG 儲存失敗') }
  }

  const deactivateGuidance = async (guidanceId: number) => {
    try {
      await api(`/api/rag/guidance/${guidanceId}`, { method: 'PATCH', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ is_active: false }) })
      showToast('RAG 建議已停用，歷史紀錄仍保留')
      setRagGuidance(await api('/api/rag/guidance?limit=20'))
    } catch (error) { showToast(error instanceof Error ? error.message : '停用失敗') }
  }

  const submitFeedback = async () => {
    if (feedbackDraft.comments.trim().length < 3) { showToast('請至少留下三個字的回饋內容'); return }
    try {
      await api('/api/professor-feedback', { method: 'POST', headers: { 'Content-Type': 'application/json' }, body: JSON.stringify({ ...feedbackDraft, page: view }) })
      setFeedbackDraft({ task: '整體教授測試', usability_score: 5, clinical_clarity_score: 5, comments: '' })
      showToast('謝謝您的回饋，已安全送出')
    } catch (error) { showToast(error instanceof Error ? error.message : '回饋送出失敗') }
  }

  if (!token || !user) return <LoginScreen login={login} setLogin={setLogin} onSubmit={handleLogin} error={loginError} />

  return <div className="shell">
    <aside className="sidebar">
      <div className="brand"><div className="brand-mark">W<span>+</span></div><div><strong>WoundCare+</strong><small>臨床照護工作台</small></div></div>
      <div className={`unit-card ${systemHealth?.status === 'ok' ? '' : 'system-warning'}`}><span className="live-dot" /> {systemHealth?.status === 'ok' ? '系統與模型已就緒' : '正在確認系統狀態'}<div className="unit-caption">{systemHealth?.professor_preview ? '教授預覽版 · 去識別化資料' : `護理站 · ${shift}`}</div></div>
      <nav className="nav">
        <NavButton active={view === 'board'} icon="⌂" label="照護總覽" onClick={() => setView('board')} />
        <NavButton active={view === 'assessment'} icon="＋" label="新增傷口評估" onClick={() => setView('assessment')} />
        <NavButton active={view === 'patients'} icon="▦" label="病人名單" onClick={() => setView('patients')} />
        {user.role !== 'nurse' && <NavButton active={view === 'audit'} icon="◷" label="覆核與稽核" onClick={() => setView('audit')} />}
        {user.role === 'admin' && <NavButton active={view === 'admin'} icon="⚙" label="系統管理" onClick={() => setView('admin')} />}
        <NavButton active={view === 'feedback'} icon="✦" label="教授回饋" onClick={() => setView('feedback')} />
      </nav>
      <div className="sidebar-foot"><div className="profile"><div className="avatar">{user.username.slice(0, 1).toUpperCase()}</div><div><strong>{user.username}</strong><small>{roleLabel[user.role]}</small></div></div><button className="logout" onClick={handleLogout}>登出工作台</button></div>
    </aside>
    <main className="main">
      <header className="topbar"><div><p className="eyebrow">{systemHealth?.professor_preview ? 'INVITATION-ONLY PROFESSOR PREVIEW' : 'WARD OPERATIONS / TODAY'}</p><h1>{view === 'board' ? '照護總覽' : view === 'assessment' ? '新增傷口評估' : view === 'patients' ? '病人名單' : view === 'audit' ? '覆核與稽核' : view === 'feedback' ? '教授測試回饋' : '系統管理'}</h1></div><div className="top-actions"><select value={shift} onChange={e => setShift(e.target.value)} aria-label="目前班別"><option>白班</option><option>小夜</option><option>大夜</option></select><span className="date-label">{new Date().toLocaleDateString('zh-TW', { month: 'long', day: 'numeric', weekday: 'short' })}</span></div></header>
      {view === 'assessment' && prediction && <section className={`preview-safety ${prediction.low_confidence || prediction.fallback_used ? 'attention' : ''}`}><div><strong>{prediction.low_confidence || prediction.fallback_used ? '需要進一步覆核' : '人工覆核仍為必要步驟'}</strong><p>{prediction.low_confidence || prediction.fallback_used ? '模型信心不足或已使用原圖 fallback；請不要直接採用 AI 分類。' : 'AI 輸出僅供決策支援，儲存前請確認傷口位置、分類與處置。'}</p>{prediction.model && <small>模型：{prediction.model.file} · SHA-256：{prediction.model.sha256?.slice(0, 12) ?? '未取得'} · 輸入 {prediction.model.input_size}px</small>}</div><label className="review-attestation"><input type="checkbox" checked={humanReviewConfirmed} onChange={e => setHumanReviewConfirmed(e.target.checked)} />我已親自覆核影像、AI 輸出與處置內容。</label></section>}
      {view === 'board' && <Board patients={filteredPatients} counts={counts} search={search} setSearch={setSearch} statusFilter={statusFilter} setStatusFilter={setStatusFilter} onSelect={(patient) => { setSelectedPatient(patient); setView('assessment'); void api(`/api/patients/${patient.id}/emr`).then(setRecords) }} onNew={() => setView('assessment')} loading={loading} />}
      {view === 'patients' && <Patients patients={filteredPatients} search={search} setSearch={setSearch} onNew={user.role === 'nurse' ? undefined : () => setView('admin')} onSelect={(patient) => { setSelectedPatient(patient); setView('assessment'); void api(`/api/patients/${patient.id}/emr`).then(setRecords) }} />}
      {view === 'assessment' && <Assessment patient={selectedPatient} patients={patients} setPatient={setSelectedPatient} imagePreview={imagePreview} prediction={prediction} fileInput={fileInput} onImage={handleImage} runAnalysis={runAnalysis} loading={loading} treatment={treatment} setTreatment={setTreatment} status={status} setStatus={setStatus} reviewStatus={reviewStatus} setReviewStatus={setReviewStatus} save={saveAssessment} records={records} />}
      {view === 'audit' && <Audit records={records} audit={audit} ragGuidance={ragGuidance} guidanceDraft={guidanceDraft} setGuidanceDraft={setGuidanceDraft} saveGuidance={saveGuidance} deactivateGuidance={deactivateGuidance} />}
      {view === 'admin' && <Admin users={users} newPatient={newPatient} setNewPatient={setNewPatient} createPatient={createPatient} newUser={newUser} setNewUser={setNewUser} createUser={createUser} />}
      {view === 'feedback' && <ProfessorFeedback draft={feedbackDraft} setDraft={setFeedbackDraft} submit={submitFeedback} health={systemHealth} />}
      {view === 'feedback' && user.role !== 'nurse' && <FeedbackArchive feedback={feedback} />}
    </main>
    {toast && <div className="toast" role="status">{toast}</div>}
  </div>
}

function LoginScreen({ login, setLogin, onSubmit, error }: { login: { username: string; password: string }; setLogin: (value: { username: string; password: string }) => void; onSubmit: (event: FormEvent) => void; error: string }) {
  return <div className="login-page">
    <section className="login-art" aria-label="WoundCare+ 臨床照護決策支援">
      <div className="art-grid" aria-hidden="true" />
      <div className="art-glow" aria-hidden="true" />
      <div className="art-copy">
        <div className="brand-mark large">W<span>+</span></div>
        <p className="eyebrow">CLINICAL DECISION SUPPORT</p>
        <h1>讓每一次<br /><em>照護判斷</em>都有依據。</h1>
        <p>把影像分析、護理評估與團隊覆核，收進同一個安靜、清楚的工作流程。</p>
        <div className="art-principles" aria-label="系統設計原則">
          <div>
            <strong>人工覆核</strong>
            <span>模型結果需由臨床人員確認</span>
          </div>
          <div>
            <strong>操作可追溯</strong>
            <span>重要決策保留稽核紀錄</span>
          </div>
        </div>
        <div className="art-note">AI 僅供決策支援 · 臨床人員保有最終判斷權</div>
      </div>
    </section>
    <main className="login-panel">
      <div className="login-card">
        <div className="login-heading">
          <p className="eyebrow">SECURE NURSE PORTAL</p>
          <h2>登入照護工作台</h2>
          <p>使用你的工作帳號，安全進入病房照護資料。</p>
        </div>
        <form onSubmit={onSubmit} className="login-form">
          <label>
            <span>工作帳號</span>
            <input
              autoComplete="username"
              required
              value={login.username}
              onChange={e => setLogin({ ...login, username: e.target.value })}
              placeholder="輸入工作帳號"
              aria-describedby={error ? 'login-error' : undefined}
            />
          </label>
          <label>
            <span>密碼</span>
            <input
              type="password"
              autoComplete="current-password"
              required
              value={login.password}
              onChange={e => setLogin({ ...login, password: e.target.value })}
              placeholder="輸入密碼"
              aria-describedby={error ? 'login-error' : undefined}
            />
          </label>
          {error && <div id="login-error" className="form-error" role="alert">{error}</div>}
          <button className="primary-button login-submit" type="submit">進入工作台 <span aria-hidden="true">→</span></button>
        </form>
        <div className="login-card-footer">
          <p className="privacy-note">受保護的工作階段 · 連線資料不會留在瀏覽器</p>
          <p className="session-note">登入後的關鍵操作會保留於系統稽核紀錄。</p>
        </div>
      </div>
    </main>
  </div>
}

function NavButton({ active, icon, label, onClick }: { active: boolean; icon: string; label: string; onClick: () => void }) { return <button className={`nav-button ${active ? 'active' : ''}`} aria-current={active ? 'page' : undefined} onClick={onClick}><span aria-hidden="true">{icon}</span>{label}</button> }

function Board({ patients, counts, search, setSearch, statusFilter, setStatusFilter, onSelect, onNew, loading }: { patients: Patient[]; counts: { total: number; red: number; yellow: number; green: number }; search: string; setSearch: (v: string) => void; statusFilter: 'all' | Status; setStatusFilter: (v: 'all' | Status) => void; onSelect: (p: Patient) => void; onNew: () => void; loading: boolean }) {
  return <div className="content-stack"><section className="hero-row"><div><p className="eyebrow">NURSING STATION / LIVE CENSUS</p><h2>今天，先照顧<br /><span>需要你的人。</span></h2><p className="muted">從病人狀態開始，快速完成影像分析與人工覆核。</p></div><button className="primary-button compact" onClick={onNew}>＋ 新增傷口評估</button></section><section className="metric-grid"><Metric label="照護中病人" value={counts.total} detail="今日病房名單" tone="teal" /><Metric label="優先處理" value={counts.red} detail="需要立即覆核" tone="coral" /><Metric label="需追蹤" value={counts.yellow} detail="等待後續評估" tone="amber" /><Metric label="穩定" value={counts.green} detail="目前無警示" tone="sage" /></section><section className="section-head"><div><p className="eyebrow">PATIENT BOARD</p><h3>病人照護看板</h3></div><div className="filters"><div className="search"><span aria-hidden="true">⌕</span><input aria-label="搜尋病人" value={search} onChange={e => setSearch(e.target.value)} placeholder="搜尋姓名、床號或病歷編號" /></div><select aria-label="依照照護狀態篩選病人" value={statusFilter} onChange={e => setStatusFilter(e.target.value as 'all' | Status)}><option value="all">全部狀態</option><option value="red">優先處理</option><option value="yellow">需追蹤</option><option value="green">穩定</option></select></div></section><section className="patient-grid">{loading ? <EmptyState text="正在同步照護資料…" /> : patients.length === 0 ? <EmptyState text="目前沒有符合條件的病人" /> : patients.map(patient => <PatientCard key={patient.id} patient={patient} onClick={() => onSelect(patient)} />)}</section></div>
}

function Metric({ label, value, detail, tone }: { label: string; value: number; detail: string; tone: string }) { return <div className={`metric-card ${tone}`}><div className="metric-icon" /><p>{label}</p><strong>{value}</strong><small>{detail}</small></div> }
function PatientCard({ patient, onClick }: { patient: Patient; onClick: () => void }) { return <button className="patient-card" onClick={onClick}><div className="patient-card-top"><span className={`status-dot ${patient.status}`} /><span className="room">{patient.room}</span><span className={`status-pill ${patient.status}`}>{statusLabel[patient.status]}</span></div><div className="patient-name">{patient.name}</div><div className="patient-meta">{patient.gender || '—'} · {patient.age || '—'} 歲 <span>#{patient.id}</span></div><div className="patient-foot"><span>最後評估</span><strong>{patient.last_assessment || '尚無紀錄'}</strong><span className="arrow">→</span></div></button> }
function Patients({ patients, search, setSearch, onNew, onSelect }: { patients: Patient[]; search: string; setSearch: (v: string) => void; onNew?: () => void; onSelect: (p: Patient) => void }) { return <div className="content-stack"><section className="section-head prominent"><div><p className="eyebrow">PATIENT DIRECTORY</p><h2>病人名單</h2><p className="muted">點選病人即可查看評估與照護歷程。</p></div>{onNew && <button className="primary-button compact" onClick={onNew}>＋ 新增病人</button>}</section><div className="search wide"><span aria-hidden="true">⌕</span><input aria-label="搜尋病人" value={search} onChange={e => setSearch(e.target.value)} placeholder="搜尋姓名、床號或病歷編號" /></div><section className="patient-grid">{patients.map(patient => <PatientCard key={patient.id} patient={patient} onClick={() => onSelect(patient)} />)}</section></div> }

function Assessment({ patient, patients, setPatient, imagePreview, prediction, fileInput, onImage, runAnalysis, loading, treatment, setTreatment, status, setStatus, reviewStatus, setReviewStatus, save, records }: { patient: Patient | null; patients: Patient[]; setPatient: (p: Patient | null) => void; imagePreview: string | null; prediction: Prediction | null; fileInput: RefObject<HTMLInputElement>; onImage: (event: ChangeEvent<HTMLInputElement>) => void; runAnalysis: () => void; loading: boolean; treatment: string; setTreatment: (v: string) => void; status: Status; setStatus: (v: Status) => void; reviewStatus: 'reviewed' | 'needs_review'; setReviewStatus: (v: 'reviewed' | 'needs_review') => void; save: () => void; records: EMRRecord[] }) {
  const rag = prediction?.rag_guidance ?? []
 return <div className="content-stack"><section className="section-head prominent"><div><p className="eyebrow">ASSESSMENT WORKFLOW / HUMAN REVIEW REQUIRED</p><h2>新增傷口評估</h2><p className="muted">先選擇病人，再上傳影像。AI 結果必須經人工確認後才能存入紀錄。</p></div></section><div className="workflow-grid"><section className="panel"><div className="panel-title"><span className="step">01</span><div><h3>選擇照護對象</h3><p>評估紀錄會與此病人綁定</p></div></div><select className="field" value={patient?.id ?? ''} onChange={e => setPatient(patients.find(item => item.id === Number(e.target.value)) ?? null)}><option value="">選擇病人或床號</option>{patients.map(item => <option key={item.id} value={item.id}>{item.room} · {item.name} · #{item.id}</option>)}</select>{patient && <div className={`selected-patient ${patient.status}`}><span className={`status-dot ${patient.status}`} /><div><strong>{patient.name}</strong><small>{patient.room} · 目前狀態：{statusLabel[patient.status]}</small></div></div>}</section><section className="panel"><div className="panel-title"><span className="step">02</span><div><h3>上傳傷口影像</h3><p>支援 JPG、PNG、WebP，最大 10 MB</p></div></div><input ref={fileInput} type="file" accept="image/jpeg,image/png,image/webp" onChange={onImage} hidden /><div className={`upload-zone ${imagePreview ? 'has-image' : ''}`} onClick={() => fileInput.current?.click()}>{imagePreview ? <img src={prediction?.image ?? imagePreview} alt="待評估傷口影像" /> : <><div className="upload-icon">↑</div><strong>拖曳或點擊上傳影像</strong><span>請確保傷口完整、光線均勻</span></>}</div>{imagePreview && <div className="upload-actions"><button className="text-button" onClick={() => fileInput.current?.click()}>更換影像</button><button className="primary-button compact" onClick={runAnalysis} disabled={loading}>{loading ? '分析中…' : '開始影像分析 →'}</button></div>}</section><section className="panel review-panel"><div className="panel-title"><span className="step">03</span><div><h3>人工覆核與處置</h3><p>AI 不能取代臨床判斷，請完成下方確認</p></div></div>{prediction ? <><div className="ai-result"><div className="result-head"><span className="result-badge">AI 分析完成</span><span>{prediction.detected_boxes} 個候選區域</span></div><div className="mode-row"><span className="secure-tag">{prediction.inference_mode === 'segmentation_crop' ? '已使用 ROI 裁切' : prediction.inference_mode === 'full_image_primary' ? '原圖分類＋ROI 標示' : prediction.inference_mode === 'full_image_fallback' ? '已回退原圖分類' : '物件偵測分類'}</span>{prediction.class_confidence != null && <span className="muted">分類信心 {Math.round(prediction.class_confidence * 100)}%</span>}</div><div className="class-row">{prediction.classes.length ? prediction.classes.map(item => <span key={item} className="class-chip">{item}</span>) : <span className="muted">未偵測到可信區域</span>}</div><p>{prediction.llm_advice}</p></div>{rag.length > 0 && <div className="rag-context"><div className="rag-context-title"><span>RAG</span><strong>護理長核准的相關建議</strong></div>{rag.map(item => <div className="rag-mini" key={item.id}><strong>{item.title}</strong><p>{item.recommendation}</p>{item.precautions && <small>注意：{item.precautions}</small>}</div>)}<small className="rag-disclaimer">僅供參考；請依病人實際狀況與院內規範覆核。</small></div>}</> : <div className="review-placeholder">完成影像分析後，這裡會顯示模型結果、RAG 建議與臨床確認欄位。</div>}<label className="field-label">護理處置與觀察紀錄<textarea className="field textarea" rows={4} value={treatment} onChange={e => setTreatment(e.target.value)} placeholder="例如：清潔傷口、覆蓋敷料、觀察滲液與周圍皮膚…" /></label><div className="choice-row"><div><span className="field-label">病人狀態</span><div className="choice-group">{(['red', 'yellow', 'green'] as Status[]).map(item => <button key={item} className={`choice ${status === item ? 'selected' : ''} ${item}`} onClick={() => setStatus(item)}>{statusLabel[item]}</button>)}</div></div><div><span className="field-label">覆核狀態</span><select className="field compact-field" value={reviewStatus} onChange={e => setReviewStatus(e.target.value as 'reviewed' | 'needs_review')}><option value="reviewed">我已完成人工確認</option><option value="needs_review">交由護理長覆核</option></select></div></div><button className="primary-button save-button" onClick={save}>儲存評估紀錄 <span>→</span></button></section></div>{records.length > 0 && <section className="panel history-panel"><div className="panel-title"><span className="step">04</span><div><h3>近期照護紀錄</h3><p>{patient?.name ?? '目前病人'} 的時間軸</p></div></div>{records.slice(0, 5).map(record => <div className="timeline-item" key={record.id}><span className={`status-dot ${record.status}`} /><div><strong>{record.treatment}</strong><small>{record.timestamp} · {record.nurse_name} · {record.shift}</small></div><span className={`status-pill ${record.status}`}>{statusLabel[record.status]}</span></div>)}</section>}</div>
}

function Audit({ records, audit, ragGuidance, guidanceDraft, setGuidanceDraft, saveGuidance, deactivateGuidance }: { records: EMRRecord[]; audit: Array<{ id: number; username?: string; action: string; resource?: string; created_at: string }>; ragGuidance: RAGGuidance[]; guidanceDraft: GuidanceDraft; setGuidanceDraft: (draft: GuidanceDraft) => void; saveGuidance: () => void; deactivateGuidance: (id: number) => void }) {
  return <div className="content-stack"><section className="section-head prominent"><div><p className="eyebrow">CLINICAL GOVERNANCE</p><h2>覆核與稽核</h2><p className="muted">查看團隊評估紀錄、操作軌跡，並將護理長核准的處置累積到 RAG。</p></div></section><section className="two-column"><div className="panel"><div className="panel-heading"><div><p className="eyebrow">ASSESSMENTS</p><h3>近期評估</h3></div><span className="count-badge">{records.length}</span></div>{records.slice(0, 10).map(record => <div className="audit-row" key={record.id}><span className={`status-dot ${record.status}`} /><div><strong>{record.patient_name ?? `病人 #${record.patient_id}`}</strong><small>{record.treatment}</small></div><div className="audit-end"><span>{record.nurse_name}</span><small>{record.timestamp}</small></div></div>)}{records.length === 0 && <EmptyState text="尚無評估紀錄" />}</div><div className="panel"><div className="panel-heading"><div><p className="eyebrow">SYSTEM TRACE</p><h3>操作稽核</h3></div><span className="secure-tag">已保護</span></div>{audit.slice(0, 12).map(item => <div className="audit-row" key={item.id}><span className="audit-icon">·</span><div><strong>{item.action}</strong><small>{item.resource || 'system'}</small></div><div className="audit-end"><span>{item.username || 'system'}</span><small>{item.created_at}</small></div></div>)}</div></section><section className="panel rag-editor"><div className="panel-heading"><div><p className="eyebrow">RAG KNOWLEDGE BASE</p><h3>新增護理長建議</h3><p className="muted">請只輸入去識別化的照護判斷與處置，不要寫入姓名、床號或病歷號。</p></div><span className="secure-tag">核准後可檢索</span></div><div className="form-grid"><label className="field-label">建議標題<input className="field" value={guidanceDraft.title} onChange={e => setGuidanceDraft({ ...guidanceDraft, title: e.target.value })} placeholder="例如：小面積擦傷的初步處置" /></label><label className="field-label">適用傷口類別<input className="field" value={guidanceDraft.wound_classes} onChange={e => setGuidanceDraft({ ...guidanceDraft, wound_classes: e.target.value })} placeholder="例如：Abrasions, Laceration" /></label><label className="field-label span-2">建議處置<textarea className="field textarea" rows={3} value={guidanceDraft.recommendation} onChange={e => setGuidanceDraft({ ...guidanceDraft, recommendation: e.target.value })} placeholder="描述護理長核准的處置步驟、觀察重點與升級條件…" /></label><label className="field-label">判斷依據<textarea className="field textarea" rows={2} value={guidanceDraft.rationale} onChange={e => setGuidanceDraft({ ...guidanceDraft, rationale: e.target.value })} placeholder="例如：依傷口深度、滲液與周圍紅腫判斷" /></label><label className="field-label">注意事項<textarea className="field textarea" rows={2} value={guidanceDraft.precautions} onChange={e => setGuidanceDraft({ ...guidanceDraft, precautions: e.target.value })} placeholder="禁忌、需通知醫師或護理長的情況" /></label><label className="field-label">對應評估紀錄<select className="field" value={guidanceDraft.source_emr_id} onChange={e => setGuidanceDraft({ ...guidanceDraft, source_emr_id: e.target.value })}><option value="">一般照護知識（無特定個案）</option>{records.map(record => <option key={record.id} value={record.id}>#{record.id} · {record.patient_name ?? `病人 #${record.patient_id}`}</option>)}</select></label><label className="field-label">證據層級<select className="field" value={guidanceDraft.evidence_level} onChange={e => setGuidanceDraft({ ...guidanceDraft, evidence_level: e.target.value })}><option value="clinical_consensus">護理長臨床共識</option><option value="protocol">院內流程</option><option value="guideline">外部指引</option></select></label></div><button className="primary-button" onClick={saveGuidance}>儲存到 RAG 知識庫 →</button></section><section className="panel"><div className="panel-heading"><div><p className="eyebrow">ACTIVE GUIDANCE</p><h3>目前可檢索建議</h3></div><span className="count-badge">{ragGuidance.length}</span></div>{ragGuidance.slice(0, 10).map(item => <div className="rag-library-row" key={item.id}><div><strong>{item.title}</strong><small>{item.wound_classes || '一般照護'} · {item.evidence_level} · {item.source}</small><p>{item.recommendation}</p></div><div className="rag-row-actions"><span className="secure-tag">啟用</span><button className="text-button danger-text" onClick={() => deactivateGuidance(item.id)}>停用</button></div></div>)}{ragGuidance.length === 0 && <EmptyState text="尚未建立護理長建議" />}</section></div>
}

function Admin({ users, newPatient, setNewPatient, createPatient, newUser, setNewUser, createUser }: { users: UserAccount[]; newPatient: { room: string; name: string; gender: string; age: string; admission_date: string; medical_history: string; allergies: string; assigned_nurse: string }; setNewPatient: (v: typeof newPatient) => void; createPatient: () => void; newUser: { username: string; password: string; role: Role }; setNewUser: (v: typeof newUser) => void; createUser: () => void }) { return <div className="content-stack"><section className="section-head prominent"><div><p className="eyebrow">ADMINISTRATION</p><h2>系統管理</h2><p className="muted">管理照護對象與團隊帳號。所有變更都會留下稽核紀錄。</p></div></section><div className="admin-grid"><section className="panel"><div className="panel-heading"><div><p className="eyebrow">NEW PATIENT</p><h3>建立病人資料</h3></div><span className="secure-tag">欄位加密</span></div><div className="form-grid"><Field label="床號" value={newPatient.room} onChange={v => setNewPatient({ ...newPatient, room: v })} /><Field label="姓名" value={newPatient.name} onChange={v => setNewPatient({ ...newPatient, name: v })} /><Field label="性別" value={newPatient.gender} onChange={v => setNewPatient({ ...newPatient, gender: v })} /><Field label="年齡" value={newPatient.age} onChange={v => setNewPatient({ ...newPatient, age: v })} /><Field label="入院日期" type="date" value={newPatient.admission_date} onChange={v => setNewPatient({ ...newPatient, admission_date: v })} /><label className="field-label span-2">指派護理師<select className="field" value={newPatient.assigned_nurse} onChange={e => setNewPatient({ ...newPatient, assigned_nurse: e.target.value })}><option value="">尚未指派（僅管理者／護理長可檢視）</option>{users.filter(item => item.role === 'nurse' && item.is_active).map(item => <option key={item.id} value={item.username}>{item.username}</option>)}</select></label><label className="field-label span-2">病史<textarea className="field textarea" rows={2} value={newPatient.medical_history} onChange={e => setNewPatient({ ...newPatient, medical_history: e.target.value })} /></label><label className="field-label span-2">過敏史<textarea className="field textarea" rows={2} value={newPatient.allergies} onChange={e => setNewPatient({ ...newPatient, allergies: e.target.value })} /></label></div><button className="primary-button" onClick={createPatient}>建立並加密存檔 →</button></section><section className="panel"><div className="panel-heading"><div><p className="eyebrow">TEAM ACCESS</p><h3>新增團隊帳號</h3></div><span className="secure-tag">RBAC</span></div><div className="form-grid"><Field label="帳號" value={newUser.username} onChange={v => setNewUser({ ...newUser, username: v })} /><Field label="密碼（至少 12 碼）" type="password" value={newUser.password} onChange={v => setNewUser({ ...newUser, password: v })} /><label className="field-label span-2">角色<select className="field" value={newUser.role} onChange={e => setNewUser({ ...newUser, role: e.target.value as Role })}><option value="nurse">臨床護理師</option><option value="head_nurse">護理長</option><option value="admin">系統管理員</option></select></label></div><button className="primary-button" onClick={createUser}>建立帳號 →</button><div className="user-list">{users.map(item => <div className="user-row" key={item.id}><div className="avatar small">{item.username.slice(0, 1).toUpperCase()}</div><div><strong>{item.username}</strong><small>{roleLabel[item.role]}</small></div><span className={item.is_active ? 'active-tag' : 'inactive-tag'}>{item.is_active ? '啟用中':'已停用'}</span></div>)}</div></section></div></div> }
function Field({ label, value, onChange, type = 'text' }: { label: string; value: string; onChange: (value: string) => void; type?: string }) { return <label className="field-label">{label}<input className="field" type={type} value={value} onChange={e => onChange(e.target.value)} /></label> }

function ProfessorFeedback({ draft, setDraft, submit, health }: { draft: { task: string; usability_score: number; clinical_clarity_score: number; comments: string }; setDraft: (draft: { task: string; usability_score: number; clinical_clarity_score: number; comments: string }) => void; submit: () => void; health: SystemHealth | null }) {
  return <div className="content-stack"><section className="section-head prominent"><div><p className="eyebrow">PROFESSOR REVIEW</p><h2>教授測試回饋</h2><p className="muted">請評估工作流程與資訊清楚度；請勿在此輸入真實病人識別資料或影像內容。</p></div></section><section className="panel feedback-panel"><div className="preview-notice"><strong>{health?.professor_preview ? '教授預覽模式已啟用' : '受保護測試模式'}</strong><p>本系統是研究原型，AI 僅供決策支援，不能作為臨床診斷或自動處置依據。</p></div><div className="form-grid"><label className="field-label span-2">測試任務<input className="field" value={draft.task} onChange={e => setDraft({ ...draft, task: e.target.value })} /></label><label className="field-label">操作易用性（1–5）<select className="field" value={draft.usability_score} onChange={e => setDraft({ ...draft, usability_score: Number(e.target.value) })}>{[1, 2, 3, 4, 5].map(score => <option key={score} value={score}>{score}</option>)}</select></label><label className="field-label">臨床資訊清楚度（1–5）<select className="field" value={draft.clinical_clarity_score} onChange={e => setDraft({ ...draft, clinical_clarity_score: Number(e.target.value) })}>{[1, 2, 3, 4, 5].map(score => <option key={score} value={score}>{score}</option>)}</select></label><label className="field-label span-2">意見與建議<textarea className="field textarea" rows={6} value={draft.comments} onChange={e => setDraft({ ...draft, comments: e.target.value })} placeholder="例如：流程中最清楚／最需要改善的地方、對研究展示的建議…" /></label></div><button className="primary-button" onClick={submit}>送出教授回饋 →</button></section></div>
}
function FeedbackArchive({ feedback }: { feedback: ProfessorFeedbackItem[] }) {
  return <section className="panel feedback-archive"><div className="panel-heading"><div><p className="eyebrow">SUBMITTED FEEDBACK</p><h3>已收回饋</h3></div><span className="count-badge">{feedback.length}</span></div>{feedback.slice(0, 10).map(item => <div className="audit-row" key={item.id}><span className="audit-icon">✦</span><div><strong>{item.task} · 易用性 {item.usability_score}/5 · 清楚度 {item.clinical_clarity_score}/5</strong><small>{item.comments}</small></div><div className="audit-end"><span>{item.username}</span><small>{item.created_at}</small></div></div>)}{feedback.length === 0 && <EmptyState text="尚未收到教授回饋" />}</section>
}
function EmptyState({ text }: { text: string }) { return <div className="empty-state">{text}</div> }

export default App
