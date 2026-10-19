import type { InferenceResult } from '../constants';
import { Clock, Cpu, Layers, Target } from 'lucide-react';

interface Props {
  result: InferenceResult;
}

function MetricRow({ label, value, sub }: { label: string; value: string; sub?: string }) {
  return (
    <tr>
      <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', opacity: 0.7 }}>{label}</td>
      <td style={{ fontWeight: 600 }}>{value}</td>
      {sub && <td style={{ fontFamily: 'var(--font-mono)', fontSize: '0.68rem', opacity: 0.55 }}>{sub}</td>}
    </tr>
  );
}

export default function ResultsPanel({ result }: Props) {
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 24 }}>
      {/* Key metrics row */}
      <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(150px, 1fr))', gap: 14 }}>
        <div className="metric-card">
          <div className="metric-card__label"><Clock size={10} /> Inference</div>
          <div className="metric-card__value">{result.inference_ms.toFixed(0)}</div>
          <div className="metric-card__sub">milliseconds</div>
        </div>
        <div className="metric-card">
          <div className="metric-card__label"><Target size={10} /> Lesion Area</div>
          <div className="metric-card__value">{result.lesion_pct.toFixed(1)}</div>
          <div className="metric-card__sub">% of pixels</div>
        </div>
        <div className="metric-card">
          <div className="metric-card__label"><Layers size={10} /> Threshold</div>
          <div className="metric-card__value">{result.threshold.toFixed(2)}</div>
          <div className="metric-card__sub">sigmoid cutoff</div>
        </div>
        <div className="metric-card">
          <div className="metric-card__label"><Cpu size={10} /> Model Input</div>
          <div className="metric-card__value" style={{ fontSize: '1.5rem' }}>224²</div>
          <div className="metric-card__sub">px resolution</div>
        </div>
      </div>

      {/* Detailed table */}
      <div className="card card-shadow" style={{ overflow: 'hidden' }}>
        <div style={{ padding: '12px 16px', background: 'var(--teal)', color: 'var(--ivory)' }}>
          <span style={{ fontFamily: 'var(--font-mono)', fontSize: '0.72rem', letterSpacing: '0.08em', textTransform: 'uppercase' }}>
            Inference Details
          </span>
        </div>
        <table className="data-table">
          <tbody>
            <MetricRow label="Input image (original)" value={`${result.input_w} × ${result.input_h} px`} />
            <MetricRow label="Model input resolution" value={`${result.model_res} × ${result.model_res} px`} />
            <MetricRow label="Predicted mask size" value={`${result.mask_w} × ${result.mask_h} px`} />
            <MetricRow label="Lesion pixel coverage" value={`${result.lesion_pct.toFixed(2)}%`} sub="not a clinical severity score" />
            <MetricRow label="Inference time" value={`${result.inference_ms.toFixed(1)} ms`} />
            <MetricRow label="Model architecture" value={result.architecture} />
            <MetricRow label="Checkpoint epoch" value={result.epoch != null ? String(result.epoch) : 'Unknown'} />
            {result.val_dice != null && (
              <MetricRow label="Checkpoint val Dice" value={result.val_dice.toFixed(4)} sub="dataset-level, not this image" />
            )}
            {result.val_iou != null && (
              <MetricRow label="Checkpoint val IoU" value={result.val_iou.toFixed(4)} sub="dataset-level, not this image" />
            )}
          </tbody>
        </table>
      </div>

      {/* Individual metrics disclaimer */}
      <p
        role="note"
        style={{
          fontFamily: 'var(--font-mono)',
          fontSize: '0.68rem',
          opacity: 0.6,
          background: 'var(--yellow)',
          border: '1.5px solid var(--ink)',
          borderRadius: '3px',
          padding: '10px 14px',
        }}
      >
        ℹ Individual Dice &amp; IoU are only shown when a ground-truth mask is provided and verified. Dataset-level
        validation metrics (Dice 0.8723, IoU 0.7916) are from the 150-image validation split — not from this image.
      </p>
    </div>
  );
}
