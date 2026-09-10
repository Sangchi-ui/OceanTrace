'use client'

import { useState, useRef } from 'react'
import {
  Activity,
  ArrowUpRight,
  CheckCircle2,
  ChevronDown,
  CircleHelp,
  CloudUpload,
  FileImage,
  Gauge,
  Layers3,
  Map,
  Play,
  ScanSearch,
  Settings2,
  ShieldCheck,
  Waves,
} from 'lucide-react'

const stages = [
  { label: 'Ingest', detail: 'GeoTIFF or PNG', icon: FileImage },
  { label: 'Normalize', detail: '1st–99th percentile', icon: Settings2 },
  { label: 'Segment', detail: 'EfficientNet-B0', icon: ScanSearch },
  { label: 'Review', detail: 'Confidence overlay', icon: Map },
]

export default function Page() {
  const [file, setFile] = useState<File | null>(null)
  const [preview, setPreview] = useState<string | null>(null)
  const [running, setRunning] = useState(false)
  const [complete, setComplete] = useState(false)
  const [results, setResults] = useState<any>(null)
  const [error, setError] = useState<string | null>(null)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files.length > 0) {
      const selectedFile = e.target.files[0]
      setFile(selectedFile)
      setPreview(URL.createObjectURL(selectedFile))
      setComplete(false)
      setResults(null)
      setError(null)
    }
  }

  const runAnalysis = async () => {
    if (!file || running) return
    setRunning(true)
    setComplete(false)
    setError(null)

    const formData = new FormData()
    formData.append('file', file)

    try {
      const response = await fetch('http://127.0.0.1:8000/predict', {
        method: 'POST',
        body: formData,
      })
      if (!response.ok) {
        const errData = await response.json()
        throw new Error(errData.detail || 'Analysis failed')
      }
      const data = await response.json()
      setResults(data)
      setComplete(true)
    } catch (err: any) {
      setError(err.message || 'Failed to connect to the model server.')
    } finally {
      setRunning(false)
    }
  }

  return (
    <main className="min-h-screen bg-background text-foreground">
      <header className="border-b border-border/70 bg-background/90">
        <div className="mx-auto flex max-w-[1440px] items-center justify-between px-6 py-5 lg:px-10">
          <div className="flex items-center gap-3">
            <div className="flex size-9 items-center justify-center rounded-full bg-primary text-primary-foreground">
              <Waves className="size-5" strokeWidth={2.2} />
            </div>
            <div>
              <p className="font-mono text-[11px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">OCEANTRACE</p>
              <p className="text-sm font-medium tracking-tight">SAR intelligence workspace</p>
            </div>
          </div>
          <div className="flex items-center gap-3 text-xs text-muted-foreground">
            <span className="hidden items-center gap-2 sm:flex"><span className="size-2 rounded-full bg-emerald-500" /> Model ready</span>
            <button className="flex items-center gap-1 rounded-full border border-border px-3 py-2 font-medium transition-colors hover:bg-accent" aria-label="Open help">
              <CircleHelp className="size-3.5" /> Help
            </button>
          </div>
        </div>
      </header>

      <div className="mx-auto max-w-[1440px] px-6 py-10 lg:px-10 lg:py-14">
        <section className="mb-10 flex flex-col justify-between gap-7 lg:flex-row lg:items-end">
          <div className="max-w-2xl">
            <div className="mb-5 flex items-center gap-2 font-mono text-[11px] font-semibold uppercase tracking-[0.18em] text-primary">
              <Activity className="size-3.5" /> Coastal monitoring / workspace 01
            </div>
            <h1 className="max-w-xl text-balance text-4xl font-semibold tracking-[-0.04em] sm:text-5xl">Detect oil spill signatures in satellite imagery.</h1>
            <p className="mt-5 max-w-lg text-pretty text-base leading-7 text-muted-foreground">Upload a two-channel SAR scene and run a calibrated segmentation pass with the OceanTrace model.</p>
          </div>
          <div className="flex items-center gap-2 self-start rounded-full border border-border bg-card px-4 py-2.5 text-xs text-muted-foreground lg:self-end">
            <ShieldCheck className="size-4 text-primary" />
            <span>Private analysis session</span>
          </div>
        </section>

        <div className="mb-8 grid gap-px overflow-hidden rounded-2xl border border-border bg-border sm:grid-cols-4">
          {stages.map((stage, index) => {
            const Icon = stage.icon
            return (
              <div key={stage.label} className="flex items-center gap-3 bg-card px-5 py-4">
                <span className={`flex size-9 shrink-0 items-center justify-center rounded-full ${index === 0 ? 'bg-primary text-primary-foreground' : 'bg-secondary text-muted-foreground'}`}>
                  <Icon className="size-4" />
                </span>
                <div>
                  <p className="text-sm font-semibold">{stage.label}</p>
                  <p className="font-mono text-[10px] uppercase tracking-wide text-muted-foreground">{stage.detail}</p>
                </div>
              </div>
            )
          })}
        </div>

        <div className="grid gap-6 lg:grid-cols-[minmax(0,1fr)_360px]">
          <div className="flex flex-col gap-6">
            <section className="rounded-2xl border border-border bg-card p-5 shadow-sm sm:p-7">
              <div className="mb-6 flex items-start justify-between gap-4">
                <div>
                  <p className="font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">Input scene</p>
                  <h2 className="mt-2 text-xl font-semibold tracking-tight">Load SAR imagery</h2>
                </div>
                <button className="flex items-center gap-1.5 text-xs font-medium text-muted-foreground hover:text-foreground" aria-label="Open input options">Options <ChevronDown className="size-3.5" /></button>
              </div>
              <input
                type="file"
                ref={fileInputRef}
                className="hidden"
                accept=".tif,.tiff,.png,.jpg,.jpeg"
                onChange={handleFileSelect}
              />
              <button
                type="button"
                onClick={() => fileInputRef.current?.click()}
                className={`group flex min-h-[310px] w-full flex-col items-center justify-center rounded-xl border border-dashed px-6 text-center transition-colors ${file ? 'border-primary/60 bg-primary/[0.04]' : 'border-border hover:border-primary/50 hover:bg-accent/40'}`}
              >
                {file ? <CheckCircle2 className="mb-5 size-12 text-primary" strokeWidth={1.5} /> : <CloudUpload className="mb-5 size-12 text-muted-foreground transition-transform group-hover:-translate-y-1" strokeWidth={1.4} />}
                <p className="text-sm font-semibold">{file ? file.name : 'Drop a SAR scene here'}</p>
                <p className="mt-2 max-w-xs text-xs leading-5 text-muted-foreground">{file ? `${(file.size / 1024 / 1024).toFixed(2)} MB · ready for analysis` : 'or click to browse · GeoTIFF, PNG, JPG up to 250 MB'}</p>
                {!file && <span className="mt-5 rounded-md bg-secondary px-3 py-2 text-xs font-medium">Choose file</span>}
              </button>
              <div className="mt-5 flex flex-wrap items-center gap-x-5 gap-y-2 text-[11px] text-muted-foreground">
                <span className="flex items-center gap-1.5"><Layers3 className="size-3.5" /> VV + VH channels</span>
                <span className="flex items-center gap-1.5"><Gauge className="size-3.5" /> 256 × 256 target</span>
                <span className="flex items-center gap-1.5"><ShieldCheck className="size-3.5" /> Processed in session</span>
              </div>
            </section>
            
            {results && (
              <section className="rounded-2xl border border-border bg-card p-5 shadow-sm sm:p-7">
                <h2 className="mb-6 text-xl font-semibold tracking-tight">Segmentation Output</h2>
                <div className="grid gap-6 sm:grid-cols-2 lg:grid-cols-3">
                  <div className="flex flex-col gap-3">
                    <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Confidence Heatmap</p>
                    <img src={results.images.heatmap} alt="Heatmap" className="aspect-square w-full rounded-lg object-cover border border-border" />
                  </div>
                  <div className="flex flex-col gap-3">
                    <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Binary Mask</p>
                    <img src={results.images.mask} alt="Mask" className="aspect-square w-full rounded-lg object-cover border border-border" />
                  </div>
                  <div className="flex flex-col gap-3">
                    <p className="text-xs font-medium uppercase tracking-wider text-muted-foreground">Result Overlay</p>
                    <img src={results.images.overlay} alt="Overlay" className="aspect-square w-full rounded-lg object-cover border border-border" />
                  </div>
                </div>
              </section>
            )}
          </div>

          <aside className="flex flex-col gap-6">
            <section className="rounded-2xl border border-border bg-card p-6">
              <div className="mb-6 flex items-center justify-between">
                <div>
                  <p className="font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">Model profile</p>
                  <h2 className="mt-2 text-lg font-semibold tracking-tight">Oil spill segmentation</h2>
                </div>
                <span className="rounded-full bg-secondary px-2.5 py-1 font-mono text-[10px] font-semibold text-muted-foreground">v0.73</span>
              </div>
              <dl className="space-y-4 text-sm">
                <div className="flex justify-between gap-4"><dt className="text-muted-foreground">Architecture</dt><dd className="font-medium">U-Net</dd></div>
                <div className="flex justify-between gap-4"><dt className="text-muted-foreground">Encoder</dt><dd className="font-medium">EfficientNet-B0</dd></div>
                <div className="flex justify-between gap-4"><dt className="text-muted-foreground">Preprocessing</dt><dd className="font-medium">Lee filter · k=3</dd></div>
                <div className="flex justify-between gap-4"><dt className="text-muted-foreground">Normalization</dt><dd className="font-medium">P1 / P99 clip</dd></div>
              </dl>
              <div className="mt-6 border-t border-border pt-5">
                <button onClick={runAnalysis} disabled={!file || running} className="flex w-full items-center justify-center gap-2 rounded-lg bg-primary px-4 py-3 text-sm font-semibold text-primary-foreground transition-opacity hover:opacity-90 disabled:cursor-not-allowed disabled:opacity-40">
                  {running ? <><Activity className="size-4 animate-pulse" /> Running segmentation</> : <><Play className="size-4 fill-current" /> {complete ? 'Run again' : 'Run analysis'}</>}
                </button>
                {complete && results && <p className="mt-3 flex items-center justify-center gap-1.5 text-xs text-emerald-600"><CheckCircle2 className="size-3.5" /> Analysis complete · {results.metrics.inference_time_ms} ms</p>}
                {error && <p className="mt-3 flex items-center justify-center gap-1.5 text-xs text-red-500">⚠️ {error}</p>}
              </div>
            </section>

            {!results ? (
              <section className="rounded-2xl bg-primary p-6 text-primary-foreground">
                <div className="flex items-start justify-between gap-4">
                  <div>
                    <p className="font-mono text-[10px] font-semibold uppercase tracking-[0.18em] opacity-65">Next step</p>
                    <p className="mt-2 text-lg font-semibold tracking-tight">Review confidence overlay</p>
                    <p className="mt-2 text-sm leading-6 opacity-70">Inspect the predicted mask and export a georeferenced result.</p>
                  </div>
                  <ArrowUpRight className="size-5 opacity-70" />
                </div>
              </section>
            ) : (
              <section className="rounded-2xl border border-border bg-card p-6">
                <div className="mb-6 flex items-center justify-between">
                  <div>
                    <p className="font-mono text-[10px] font-semibold uppercase tracking-[0.18em] text-muted-foreground">Results</p>
                    <h2 className="mt-2 text-lg font-semibold tracking-tight">Analysis Metrics</h2>
                  </div>
                </div>
                <dl className="space-y-4 text-sm">
                  <div className="flex justify-between gap-4"><dt className="text-muted-foreground">Oil Coverage</dt><dd className="font-medium text-primary">{results.metrics.oil_coverage_percent.toFixed(2)}%</dd></div>
                  <div className="flex justify-between gap-4"><dt className="text-muted-foreground">Mean Confidence</dt><dd className="font-medium">{(results.metrics.mean_confidence * 100).toFixed(1)}%</dd></div>
                  <div className="flex justify-between gap-4"><dt className="text-muted-foreground">Detected Pixels</dt><dd className="font-medium">{results.metrics.oil_pixels.toLocaleString()}</dd></div>
                  <div className="flex justify-between gap-4"><dt className="text-muted-foreground">Acquisition Time</dt><dd className="font-medium">{results.metadata?.acquisition_time || 'N/A'}</dd></div>
                </dl>
              </section>
            )}
          </aside>
        </div>
      </div>
    </main>
  )
}
