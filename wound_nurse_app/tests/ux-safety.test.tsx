import { test } from 'node:test'
import assert from 'node:assert/strict'
import { renderToStaticMarkup } from 'react-dom/server'
import { ResearchCareDisplay } from '../src/ResearchCareDisplay'
import { File } from 'node:buffer'
import { createUploadSession } from '../src/uploadSession'
import { UploadStatusDisplay } from '../src/UploadStatusDisplay'
import { PredictionV2Panel } from '../src/PredictionV2Panel'
import './live-v2.test'

function harness(validateImage: (url: string, signal: AbortSignal) => Promise<void> = async () => {}) {
  const revoked: string[]=[]; let url=0
  const session=createUploadSession<{label:string}>({createObjectURL:()=>`blob:test-${++url}`,revokeObjectURL:u=>revoked.push(u),validateImage})
  return {session,revoked}
}
const png=(name='one.png') => new File(['fixture'],name,{type:'image/png',lastModified:100})
function deferred<T>() {
  let resolve!: (value:T)=>void
  let reject!: (error:Error)=>void
  const promise=new Promise<T>((yes,no)=>{resolve=yes;reject=no})
  return {promise,resolve,reject}
}

test('unreviewed V1 advice and RAG text are not presented as care recommendations', () => {
  const payload = { llm_advice: 'LEGACY_UNREVIEWED_CARE', rag_guidance: [{ id:1, title:'OLD', recommendation:'UNREVIEWED_RAG' }] }
  const original = JSON.stringify(payload)
  const html = renderToStaticMarkup(<ResearchCareDisplay prediction={payload} />)
  assert.match(html, /目前尚未配置經審定的照護規則/)
  assert.doesNotMatch(html, /LEGACY_UNREVIEWED_CARE|UNREVIEWED_RAG/)
  assert.equal(JSON.stringify(payload), original)
})

test('invalid new selection clears the old classification, mask association and original preview', async () => {
  const {session}=harness()
  await session.select(png())
  await session.run(async()=>({label:'OLD_RESULT'}))
  assert.equal(session.getSnapshot().result?.label,'OLD_RESULT')
  await session.select(new File(['bad'],'wrong.txt',{type:'text/plain'}))
  assert.equal(session.getSnapshot().status,'error')
  assert.equal(session.getSnapshot().result,null)
  assert.equal(session.getSnapshot().input,null)
  assert.equal(session.getSnapshot().previewUrl,null)
})

test('a late earlier request cannot overwrite a newer input success even if transport ignores abort', async () => {
  const {session}=harness();const old=deferred<{label:string}>()
  await session.select(png('A.png'))
  const oldRun=session.run(()=>old.promise)
  const oldIdentity=session.getSnapshot()
  await session.select(png('B.png'))
  await session.run(async()=>({label:'NEW_RESULT'}))
  const next=session.getSnapshot()
  assert.notEqual(next.input?.id,oldIdentity.input?.id)
  assert.notEqual(next.requestId,oldIdentity.requestId)
  old.resolve({label:'OLD_RESULT'})
  assert.equal(await oldRun,false)
  assert.equal(session.getSnapshot().result?.label,'NEW_RESULT')
  assert.equal(session.getSnapshot().input?.name,'B.png')
})

test('a corrupt image with valid MIME fails decode without retaining any earlier result or preview', async () => {
  const {session}=harness(async()=>{throw new Error('無法解碼影像')})
  await session.select(png('corrupt.png'))
  assert.equal(session.getSnapshot().status,'error')
  assert.equal(session.getSnapshot().input,null)
  assert.equal(session.getSnapshot().result,null)
  assert.equal(session.getSnapshot().previewUrl,null)
  let sent=false
  await session.run(async()=>{sent=true;return {label:'SHOULD_NOT_SEND'}})
  assert.equal(sent,false)
})

test('retry loading and failed upload never display the previous success or its mask', async () => {
  const {session}=harness()
  await session.select(png())
  await session.run(async()=>({label:'OLD_MASK_AND_CLASS'}))
  const transport=deferred<{label:string}>()
  const pending=session.run(()=>transport.promise)
  assert.equal(session.getSnapshot().status,'loading')
  assert.equal(session.getSnapshot().result,null)
  transport.reject(new Error('HTTP upload failed'))
  await pending
  assert.equal(session.getSnapshot().status,'error')
  assert.equal(session.getSnapshot().result,null)
  assert.equal(session.getSnapshot().error,'HTTP upload failed')
})

test('object URLs are revoked once when replaced, rejected or disposed, never leaked or reused', async () => {
  const {session,revoked}=harness()
  await session.select(png('A.png'))
  await session.select(png('B.png'))
  assert.deepEqual(revoked,['blob:test-1'])
  await session.select(new File(['bad'],'wrong.txt',{type:'text/plain'}))
  assert.deepEqual(revoked,['blob:test-1','blob:test-2'])
  await session.select(png('C.png'))
  session.dispose();session.dispose()
  assert.deepEqual(revoked,['blob:test-1','blob:test-2','blob:test-3'])
})

test('pending validation cannot start an upload before decode succeeds', async () => {
  const decode=deferred<void>(); const {session}=harness(()=>decode.promise)
  const selecting=session.select(png())
  assert.equal(session.getSnapshot().status,'validating')
  let sent=false
  assert.equal(await session.run(async()=>{sent=true;return {label:'EARLY'}}),false)
  assert.equal(sent,false)
  decode.resolve();await selecting
  assert.equal(session.getSnapshot().status,'ready')
})

test('NOT_CONFIGURED V2 explicitly limits high confidence without changing its raw payload', () => {
  const payload={care_mapping_status:'NOT_CONFIGURED',llm_advice:'V2_LEGACY_TREATMENT',rag_guidance:[]}
  const before=JSON.stringify(payload)
  const html=renderToStaticMarkup(<ResearchCareDisplay prediction={payload} />)
  assert.match(html,/目前尚未配置經審定的照護規則/)
  assert.match(html,/研究展示與人工覆核/)
  assert.match(html,/NOT_CONFIGURED/)
  assert.match(html,/高 confidence 不等於診斷確認/)
  assert.match(html,/OOD 拒判能力尚未驗證/)
  assert.doesNotMatch(html,/V2_LEGACY_TREATMENT/)
  assert.equal(JSON.stringify(payload),before)
})

test('a late error after an invalid new selection cannot erase the current upload error', async () => {
  const {session}=harness();const old=deferred<{label:string}>();let signal:AbortSignal|undefined
  await session.select(png())
  const pending=session.run((_file,s)=>{signal=s;return old.promise})
  await session.select(new File(['bad'],'invalid.txt',{type:'text/plain'}))
  assert.equal(signal?.aborted,true)
  const expected=session.getSnapshot()
  old.reject(new Error('STALE_SERVER_ERROR'));assert.equal(await pending,false)
  assert.equal(session.getSnapshot(),expected)
})

test('validation completion from an earlier file never marks a later input ready', async () => {
  const old=deferred<void>()
  const {session,revoked}=harness(url=>url==='blob:test-1'?old.promise:Promise.resolve())
  const pending=session.select(png('old.png'))
  await session.select(png('new.png'))
  old.resolve();await pending
  assert.equal(session.getSnapshot().input?.name,'new.png')
  assert.equal(session.getSnapshot().status,'ready')
  assert.deepEqual(revoked,['blob:test-1'])
})

test('latest request wins even for the same input and resets clear late completion on logout/unmount', async () => {
  const {session}=harness();const older=deferred<{label:string}>()
  await session.select(png())
  const pending=session.run(()=>older.promise)
  await session.run(async()=>({label:'LATEST_SAME_INPUT'}))
  older.resolve({label:'STALE'});assert.equal(await pending,false)
  assert.equal(session.getSnapshot().result?.label,'LATEST_SAME_INPUT')
  const late=deferred<{label:string}>();const second=session.run(()=>late.promise)
  session.reset();late.resolve({label:'POST_LOGOUT'});assert.equal(await second,false)
  assert.equal(session.getSnapshot().status,'idle');assert.equal(session.getSnapshot().result,null)
})

test('empty, oversize and mismatched MIME files never reach the predict boundary', async () => {
  for(const file of [new File([],'empty.png',{type:'image/png'}),new File([new Uint8Array(10*1024*1024+1)],'huge.png',{type:'image/png'}),new File(['x'],'fake.png',{type:'application/octet-stream'})]) {
    const {session}=harness();await session.select(file)
    let sent=false;await session.run(async()=>{sent=true;return {label:'WRONG'}})
    assert.equal(sent,false);assert.equal(session.getSnapshot().status,'error')
    assert.equal(session.getSnapshot().result,null)
  }
})

test('observable loading, error and success states are separated and identity remains inspectable', async () => {
  const {session}=harness();const waiting=deferred<{label:string}>()
  await session.select(png('source.png'))
  const pending=session.run(()=>waiting.promise)
  const loading=renderToStaticMarkup(<UploadStatusDisplay snapshot={session.getSnapshot()} />)
  assert.match(loading,/data-upload-status="loading"/)
  assert.match(loading,/影像分析中/);assert.match(loading,/source.png/);assert.match(loading,/request-1/)
  assert.doesNotMatch(loading,/AI 分析完成/)
  waiting.resolve({label:'SYNTHETIC_RESULT'});await pending
  const success=renderToStaticMarkup(<UploadStatusDisplay snapshot={session.getSnapshot()} />)
  assert.match(success,/data-upload-status="success"/);assert.match(success,/結果僅屬於此輸入與請求/)
  assert.doesNotMatch(success,/role="alert"|影像分析中/)
})

test('empty V2 masks do not generate polygons or turn a merged ROI into an instance', () => {
  const html=renderToStaticMarkup(<PredictionV2Panel originalUrl="blob:synthetic" prediction={{image:'fixture',classes:[],requires_human_review:true,
    schema_version:'2.0.0',original_image_size:{width:512,height:512},segmentation_instance_count:0,segmentation_instances:[],
    predicted_bounding_boxes:[],merged_roi:{x1:1,y1:1,x2:10,y2:10}}} />)
  assert.match(html,/0 個模型分割實例/);assert.match(html,/沒有模型分割實例/)
  assert.doesNotMatch(html,/<polygon\b|<rect\b/)
})

test('browser preview allocation errors become a clear error state, not an unhandled selection failure', async () => {
  const session=createUploadSession<never>({createObjectURL:()=>{throw new Error('Preview unavailable')},revokeObjectURL:()=>{},validateImage:async()=>{}})
  await session.select(png())
  assert.equal(session.getSnapshot().status,'error')
  assert.equal(session.getSnapshot().error,'Preview unavailable')
  assert.equal(session.getSnapshot().result,null)
})
