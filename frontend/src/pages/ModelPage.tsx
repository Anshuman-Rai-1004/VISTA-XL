import ArchitectureDiagram from '../components/ArchitectureDiagram';
import { MODEL_INFO, VALIDATION_METRICS } from '../constants';
import { Cpu, GitBranch, Database, Settings2 } from 'lucide-react';

function InfoRow({ label, value }: { label: string; value: string }) {
  return (
    <tr>
      <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', opacity: 0.65 }}>{label}</td>
      <td style={{ fontWeight: 600, fontSize: '0.875rem' }}>{value}</td>
    </tr>
  );
}

export default function ModelPage() {
  const metricEntries = Object.values(VALIDATION_METRICS);

  return (
    <main className="page-content">
      <div className="container" style={{ paddingTop: 48, paddingBottom: 80 }}>

        {/* Header */}
        <div style={{ marginBottom: 48 }}>
          <div className="section-eyebrow">Model Information</div>
          <h1 style={{ fontSize: 'clamp(2rem, 5vw, 3.2rem)', marginBottom: 12 }}>
            VISTA-XL Architecture
          </h1>
          <p style={{ maxWidth: 620, opacity: 0.75 }}>
            Vision xLSTM-Enhanced Swin-UNet. A Swin-Tiny encoder with an mLSTM bottleneck
            applied to the deepest feature map, paired with a U-Net convolutional decoder.
          </p>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 40 }}>

          {/* ── Left: Architecture diagram ── */}
          <div>
            <div className="section-eyebrow" style={{ marginBottom: 16 }}>Forward Pass</div>
            <div className="card card-shadow" style={{ padding: '0 0 16px 0', overflow: 'hidden' }}>
              <div style={{
                background: 'var(--ink)', color: 'var(--ivory)',
                padding: '10px 20px',
                fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
                letterSpacing: '0.08em', textTransform: 'uppercase',
              }}>
                Model Architecture Diagram
              </div>
              <ArchitectureDiagram />
            </div>
          </div>

          {/* ── Right: Specs ── */}
          <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>

            {/* Config table */}
            <div className="card card-shadow" style={{ overflow: 'hidden' }}>
              <div style={{
                background: 'var(--teal)', color: 'var(--ivory)',
                padding: '10px 16px',
                fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
                letterSpacing: '0.08em', textTransform: 'uppercase',
                display: 'flex', alignItems: 'center', gap: 8,
              }}>
                <Cpu size={13} /> Model Configuration
              </div>
              <table className="data-table">
                <tbody>
                  <InfoRow label="Model name" value={MODEL_INFO.fullName} />
                  <InfoRow label="Encoder" value={MODEL_INFO.encoder} />
                  <InfoRow label="Bottleneck" value={MODEL_INFO.bottleneck} />
                  <InfoRow label="Decoder" value={MODEL_INFO.decoder} />
                  <InfoRow label="Input resolution" value={MODEL_INFO.inputRes} />
                  <InfoRow label="Output channels" value={String(MODEL_INFO.outputChannels)} />
                  <InfoRow label="Threshold" value={String(MODEL_INFO.threshold)} />
                  <InfoRow label="Normalization" value={MODEL_INFO.normalization} />
                </tbody>
              </table>
            </div>

            {/* Training config */}
            <div className="card card-shadow" style={{ overflow: 'hidden' }}>
              <div style={{
                background: 'var(--ink)', color: 'var(--ivory)',
                padding: '10px 16px',
                fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
                letterSpacing: '0.08em', textTransform: 'uppercase',
                display: 'flex', alignItems: 'center', gap: 8,
              }}>
                <Settings2 size={13} /> Training Setup
              </div>
              <table className="data-table">
                <tbody>
                  <InfoRow label="Dataset" value={MODEL_INFO.dataset} />
                  <InfoRow label="Loss function" value={MODEL_INFO.loss} />
                  <InfoRow label="Optimizer" value={MODEL_INFO.optimizer} />
                  <InfoRow label="Epochs" value={String(MODEL_INFO.epochs)} />
                  <InfoRow label="Batch size" value={String(MODEL_INFO.batchSize)} />
                </tbody>
              </table>
            </div>
          </div>
        </div>

        {/* ── Validation metrics section ── */}
        <div style={{ marginTop: 56 }}>
          <div className="section-eyebrow" style={{ marginBottom: 8 }}>
            <Database size={12} /> Validation Results
          </div>
          <h2 style={{ marginBottom: 8 }}>Dataset-Level Performance</h2>
          <p style={{ opacity: 0.7, marginBottom: 32, fontSize: '0.9rem', maxWidth: 620 }}>
            Evaluated on the 150-image validation split. These are dataset-level aggregate metrics —
            individual per-image Dice / IoU varies and requires a ground-truth mask.
          </p>

          <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(180px, 1fr))', gap: 16 }}>
            {metricEntries.map(m => (
              <div key={m.label} className="metric-card">
                <div className="metric-card__label">{m.label}</div>
                <div className="metric-card__value">
                  {(m.value * 100).toFixed(2)}
                  <span style={{ fontSize: '0.9rem', fontWeight: 400 }}>%</span>
                </div>
                <div className="metric-card__sub">±{(m.std * 100).toFixed(1)}% SD</div>
                <div className="metric-card__sub">{m.description}</div>
              </div>
            ))}
          </div>

          <div style={{
            marginTop: 24,
            padding: '12px 16px',
            background: 'var(--yellow)',
            border: 'var(--border)',
            borderRadius: 'var(--radius)',
            fontFamily: 'var(--font-mono)',
            fontSize: '0.7rem',
            maxWidth: 720,
          }}>
            ℹ These results are from validation, not an independent test set.
            No test set (e.g. official ISIC 2017 600-image set) is currently available locally.
            Results should not be interpreted as final research claims.
          </div>
        </div>

        {/* ── Checkpoint details ── */}
        <div style={{ marginTop: 56 }}>
          <div className="section-eyebrow" style={{ marginBottom: 16 }}>
            <GitBranch size={12} /> Checkpoint
          </div>
          <div className="card card-shadow" style={{ overflow: 'hidden', maxWidth: 520 }}>
            <div style={{
              background: 'var(--coral)', color: 'var(--ink)',
              padding: '10px 16px',
              fontFamily: 'var(--font-mono)', fontSize: '0.7rem',
              letterSpacing: '0.08em', textTransform: 'uppercase',
            }}>
              vista_xl_best.pth · 423 MB
            </div>
            <table className="data-table">
              <tbody>
                <InfoRow label="File" value="checkpoints/vista_xl_best.pth" />
                <InfoRow label="Selection criterion" value="Best validation Dice" />
                <InfoRow label="Loading" value="strict=True (all keys must match)" />
                <InfoRow label="Device" value="CPU (no CUDA GPU available)" />
                <InfoRow label="Load strategy" value="Initialized once at server startup" />
              </tbody>
            </table>
          </div>
        </div>

      </div>
    </main>
  );
}
