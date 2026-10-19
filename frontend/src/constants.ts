// API base â€“ backend FastAPI runs on 8000
export const API_BASE = '';

export interface InferenceResult {
  mask_b64: string;       // PNG base64
  overlay_b64: string;    // PNG base64
  input_w: number;
  input_h: number;
  model_res: number;
  mask_w: number;
  mask_h: number;
  lesion_pct: number;
  inference_ms: number;
  epoch: number | null;
  val_dice: number | null;
  val_iou: number | null;
  threshold: number;
  architecture: string;
}

export type AppStatus = 'idle' | 'loading' | 'success' | 'error';

export const MODEL_INFO = {
  name: 'VISTA-XL',
  fullName: 'Vision xLSTM-Enhanced Swin-UNet',
  encoder: 'Swin-Tiny (swin_tiny_patch4_window7_224)',
  bottleneck: 'mLSTM Block (embedding=768, heads=4, context=49)',
  decoder: 'U-Net-style Convolutional Decoder',
  inputRes: '224 Ã— 224 px',
  outputChannels: 1,
  threshold: 0.5,
  normalization: 'ImageNet (Î¼=[0.485,0.456,0.406], Ïƒ=[0.229,0.224,0.225])',
  loss: 'BCE + Soft Dice',
  optimizer: 'AdamW (lr=1e-4, wd=1e-5)',
  epochs: 10,
  batchSize: 8,
  dataset: 'ISIC 2017 (2,000 train / 150 val)',
} as const;

export const VALIDATION_METRICS = {
  dice: { value: 0.8723, std: 0.1232, label: 'Dice', description: 'Mean Â± SD over 150 val images' },
  iou: { value: 0.7916, std: 0.1681, label: 'IoU', description: 'Jaccard index' },
  precision: { value: 0.8969, std: 0.1332, label: 'Precision', description: 'PPV' },
  recall: { value: 0.8878, std: 0.1677, label: 'Recall', description: 'Sensitivity / TPR' },
  specificity: { value: 0.9773, std: 0.0572, label: 'Specificity', description: 'TNR' },
  pixelAcc: { value: 0.9597, std: 0.0718, label: 'Pixel Acc', description: 'Overall pixel accuracy' },
} as const;
