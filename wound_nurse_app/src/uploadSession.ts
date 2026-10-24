export type UploadStatus = 'idle' | 'validating' | 'ready' | 'loading' | 'success' | 'error'
export interface UploadInput { id: string; file: File; name: string; type: string; size: number; lastModified: number }
export interface UploadSnapshot<T> {
  status: UploadStatus
  input: UploadInput | null
  requestId: string | null
  previewUrl: string | null
  result: T | null
  error: string | null
}
export interface UploadPlatform {
  createObjectURL: (file: File) => string
  revokeObjectURL: (url: string) => void
  validateImage: (url: string, signal: AbortSignal) => Promise<void>
}

/** The frontend upload/request boundary. IDs are local generations, not content hashes. */
export function createUploadSession<T>(platform: UploadPlatform) {
  let generation = 0
  let request = 0
  let active: AbortController | null = null
  let state: UploadSnapshot<T> = { status:'idle', input:null, requestId:null, previewUrl:null, result:null, error:null }
  const listeners = new Set<() => void>()
  const publish = (next: UploadSnapshot<T>) => { state=next; listeners.forEach(fn=>fn()) }
  const reset = () => {
    generation++; active?.abort(); active=null
    if (state.previewUrl) platform.revokeObjectURL(state.previewUrl)
    publish({status:'idle',input:null,requestId:null,previewUrl:null,result:null,error:null})
  }
  return {
    getSnapshot: () => state,
    subscribe: (fn: () => void) => { listeners.add(fn); return () => { listeners.delete(fn) } },
    reset,
    dispose: reset,
    async select(file: File) {
      reset()
      if (!['image/jpeg','image/png','image/webp'].includes(file.type) || file.size===0 || file.size>10*1024*1024) {
        publish({...state,status:'error',error:'請選擇 10 MB 以內且非空的 JPG、PNG 或 WebP 圖片'}); return
      }
      const input={id:`input-${++generation}`,file,name:file.name,type:file.type,size:file.size,lastModified:file.lastModified}
      const revision=generation
      const controller=new AbortController(); active=controller
      let previewUrl: string | null = null
      try {
        previewUrl=platform.createObjectURL(file)
        publish({status:'validating',input,requestId:null,previewUrl,result:null,error:null})
        await platform.validateImage(previewUrl,controller.signal)
        if (revision!==generation || controller.signal.aborted) return
        publish({...state,status:'ready'})
      } catch (error) {
        if (revision!==generation || controller.signal.aborted) return
        if (previewUrl) platform.revokeObjectURL(previewUrl)
        publish({status:'error',input:null,requestId:null,previewUrl:null,result:null,error:error instanceof Error?error.message:'無法解碼影像'})
      }
    },
    async run(predict: (file: File, signal: AbortSignal) => Promise<T>) {
      const input=state.input
      if (!input || state.status==='validating') return false
      const revision=generation
      active?.abort()
      const controller=new AbortController(); active=controller
      const requestId=`request-${++request}`
      const current=()=>generation===revision && state.requestId===requestId && !controller.signal.aborted
      publish({...state,status:'loading',requestId,result:null,error:null})
      try {
        const result=await predict(input.file,controller.signal)
        if (!current()) return false
        publish({...state,status:'success',result,error:null}); return true
      } catch (error) {
        if (!current()) return false
        publish({...state,status:'error',result:null,error:error instanceof Error?error.message:'影像分析失敗'}); return true
      }
    },
  }
}
