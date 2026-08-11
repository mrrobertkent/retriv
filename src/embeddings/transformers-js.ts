import type { EmbeddingConfig, EmbeddingProvider, ResolvedEmbedding } from '../types'
import { rm } from 'node:fs/promises'
import { env, pipeline } from '@huggingface/transformers'
import { getModelDimensions, getModelMaxTokens, resolveModelForPreset } from './model-info'

export interface TransformersProgressInfo {
  status: 'initiate' | 'download' | 'progress' | 'done' | 'ready'
  name: string
  file: string
  progress?: number
  loaded?: number
  total?: number
}

/** Execution device supported by Transformers.js */
export type TransformersDevice
  = | 'auto'
    | 'cpu'
    | 'gpu'
    | 'wasm'
    | 'webgpu'
    | 'cuda'
    | 'dml'
    | 'coreml'
    | 'webnn'
    | 'webnn-npu'
    | 'webnn-gpu'
    | (string & {})

/** Quantization level supported by Transformers.js */
export type TransformersDtype
  = | 'auto'
    | 'fp32'
    | 'fp16'
    | 'q8'
    | 'int8'
    | 'uint8'
    | 'q4'
    | 'bnb4'
    | 'q4f16'
    | (string & {})

export interface TransformersEmbeddingOptions {
  /** Model name (e.g., 'bge-base-en-v1.5' or 'Xenova/bge-base-en-v1.5') */
  model?: string
  /** Embedding dimensions (auto-detected for known models) */
  dimensions?: number
  /**
   * Execution device. Defaults to the Transformers.js default (CPU in Node).
   * Set `'coreml'` on Apple Silicon or `'webgpu'` where available to offload
   * inference from the CPU.
   */
  device?: TransformersDevice
  /**
   * Quantization level (default: `'fp32'`). Lower precision such as `'q8'`
   * reduces model size and speeds up inference at some cost to accuracy.
   */
  dtype?: TransformersDtype
  /** Called with model download progress (initiate → download → progress → done → ready) */
  onProgress?: (info: TransformersProgressInfo) => void
}

/**
 * Clear corrupted model cache, returns true if cleared and retry should be attempted
 */
async function clearCorruptedCache(error: unknown, model: string): Promise<boolean> {
  const isProtobufError = error instanceof Error
    && (error.message?.includes('Protobuf parsing failed') || String(error.cause)?.includes('Protobuf parsing failed'))

  if (!isProtobufError || !env.cacheDir)
    return false

  const modelPath = `${env.cacheDir}/${model}`
  await rm(modelPath, { recursive: true, force: true }).catch(() => {})
  console.warn(`[retriv] Cleared corrupted model cache for ${model}, retrying...`)
  return true
}

/**
 * Transformers.js embedding provider (local, in-browser compatible)
 *
 * @example
 * ```ts
 * import { transformersJs } from 'retriv/embeddings/transformers-js'
 * import { sqliteVec } from 'retriv/db/sqlite-vec'
 *
 * // Auto-resolves model name and dimensions for known models
 * const db = await sqliteVec({
 *   path: 'vectors.db',
 *   embeddings: transformersJs({ model: 'bge-base-en-v1.5' }),
 * })
 *
 * // Offload inference to the GPU / Neural Engine on Apple Silicon
 * const fast = await sqliteVec({
 *   path: 'vectors.db',
 *   embeddings: transformersJs({ model: 'bge-base-en-v1.5', device: 'coreml', dtype: 'q8' }),
 * })
 * ```
 */
export function transformersJs(options: TransformersEmbeddingOptions = {}): EmbeddingConfig {
  const baseModel = options.model ?? 'bge-small-en-v1.5'
  const model = resolveModelForPreset(baseModel, 'transformers.js')
  let cached: ResolvedEmbedding | null = null

  return {
    async resolve() {
      if (cached)
        return cached

      const pipelineOpts: Record<string, unknown> = { dtype: options.dtype ?? 'fp32' }
      if (options.device)
        pipelineOpts.device = options.device
      if (options.onProgress)
        pipelineOpts.progress_callback = options.onProgress

      const extractor = await pipeline('feature-extraction', model, pipelineOpts)
        .catch(async (err) => {
          if (await clearCorruptedCache(err, model))
            return pipeline('feature-extraction', model, pipelineOpts)
          throw err
        })

      // Known models resolve from the registry; anything else is probed with a
      // single embedding, matching how the Ollama provider handles unknown
      // models. Without this, any Hugging Face repo outside the registry is
      // unusable even though the pipeline loads fine.
      let dimensions = options.dimensions ?? getModelDimensions(model)
      if (!dimensions) {
        const probe = await extractor(['dimension probe'], { pooling: 'mean', normalize: true })
        dimensions = (probe.data as Float32Array).length
      }
      if (!dimensions)
        throw new Error(`Could not determine dimensions for model ${model}. Please specify the dimensions option.`)

      const embedder: EmbeddingProvider = async (texts) => {
        const output = await extractor(texts, { pooling: 'mean', normalize: true })
        const data = output.data as Float32Array
        const results: Float32Array[] = Array.from({ length: texts.length })
        for (let i = 0; i < texts.length; i++)
          results[i] = data.slice(i * dimensions, (i + 1) * dimensions)
        return results
      }

      cached = { embedder, dimensions, maxTokens: getModelMaxTokens(model) }
      return cached
    },
  }
}
