import { useEffect, useState, useSyncExternalStore } from 'react'
import { createUploadSession } from './uploadSession'

/** Browser decode is a validity check only; no model input resizing or re-encoding. */
export function validateBrowserImage(url: string, signal: AbortSignal): Promise<void> {
  return new Promise((resolve,reject) => {
    const image = new Image()
    const cleanup = () => { image.onload=null; image.onerror=null; signal.removeEventListener('abort',abort) }
    const abort = () => { cleanup(); image.src=''; reject(new DOMException('已取消舊影像檢查','AbortError')) }
    if (signal.aborted) { abort(); return }
    image.onload = () => {
      const valid=image.naturalWidth>0 && image.naturalHeight>0
      cleanup()
      if (valid) resolve(); else reject(new Error('影像尺寸無效，請選擇其他圖片'))
    }
    image.onerror = () => { cleanup(); reject(new Error('無法解碼影像，請選擇完整的 JPG、PNG 或 WebP 圖片')) }
    signal.addEventListener('abort',abort,{once:true})
    image.src=url
  })
}

export function useUploadSession<T>() {
  const [session]=useState(()=>createUploadSession<T>({
    createObjectURL:file=>URL.createObjectURL(file),
    revokeObjectURL:url=>URL.revokeObjectURL(url),
    validateImage:validateBrowserImage,
  }))
  const snapshot=useSyncExternalStore(session.subscribe,session.getSnapshot,session.getSnapshot)
  useEffect(()=>()=>session.dispose(),[session])
  return {session,snapshot}
}
