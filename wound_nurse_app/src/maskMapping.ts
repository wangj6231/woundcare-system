import type { ImageSize, NativeMaskRLE } from './predictionV2'

const maxDisplayPixels = 16_777_216
function pixels(width: number, height: number) {
  if (!Number.isSafeInteger(width) || !Number.isSafeInteger(height) || width <= 0 || height <= 0 || width * height > maxDisplayPixels) throw new Error('遮罩或原圖尺寸超出檢視限制')
  return width * height
}

export function decodeNativeMask(rle: NativeMaskRLE): Uint8Array {
  if (rle.encoding !== 'BINARY_RLE_C_ORDER' || rle.coordinate_system !== 'NATIVE_RESULT_MASK_GRID' || rle.starts_with !== 0 || rle.size.length !== 2) throw new Error('不支援的原生遮罩編碼')
  const length = pixels(rle.size[1], rle.size[0])
  let sum = 0
  for (const count of rle.counts) {
    if (!Number.isSafeInteger(count) || count < 0) throw new Error('RLE run 無效')
    sum += count
    if (sum > length) throw new Error('RLE 超出原生網格')
  }
  if (sum !== length) throw new Error('RLE 並未完整覆蓋原生網格')
  const result = new Uint8Array(length)
  let offset = 0
  rle.counts.forEach((count, run) => { if (run % 2 === 1) result.fill(1, offset, offset + count); offset += count })
  return result
}

export function mapNativeMaskToOriginal(rle: NativeMaskRLE, original: ImageSize): Uint8Array {
  const native = decodeNativeMask(rle)
  const [nativeHeight, nativeWidth] = rle.size
  const result = new Uint8Array(pixels(original.width, original.height))
  // Invert the installed Ultralytics 8.3.53 centered-letterbox coordinate contract.
  // Display-only nearest-neighbour sampling; native RLE remains untouched.
  const gain = Math.min(nativeWidth / original.width, nativeHeight / original.height)
  const padX = (nativeWidth - original.width * gain) / 2
  const padY = (nativeHeight - original.height * gain) / 2
  for (let y = 0; y < original.height; y++) {
    const ny = Math.floor((y + 0.5) * gain + padY)
    for (let x = 0; x < original.width; x++) {
      const nx = Math.floor((x + 0.5) * gain + padX)
      if (nx >= 0 && nx < nativeWidth && ny >= 0 && ny < nativeHeight) result[y * original.width + x] = native[ny * nativeWidth + nx]
    }
  }
  return result
}
