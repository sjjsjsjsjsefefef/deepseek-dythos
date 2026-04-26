#!/usr/bin/env python3
"""
Native MP4 inference with 60/120fps 1-hour streaming support.
Usage:
    python video_inference.py --variant pro --mp4 path/to/video.mp4 --prompt "Describe this video"
"""
import argparse
import torch
import torch.nn.functional as F
from transformers import AutoTokenizer
from dythos import DythosConfig, DeepSeekDythos, DythosInferenceEngine


def stream_encode_mp4(model, mp4_path, max_hours=1.0):
    from torchvision.io import VideoReader
    import torchaudio.transforms as T_audio

    device = next(model.parameters()).device
    reader = VideoReader(mp4_path, "video")
    meta = reader.get_metadata()
    fps = float(meta["video"]["fps"][0])
    duration = float(meta["video"]["duration"][0])
    print(f"📹 {mp4_path} | {duration/60:.1f} min @ {fps:.0f} fps")

    frames_per_sec = int(round(fps))
    vid_enc = model.video_encoder
    all_video_tokens = []
    sec_buffer = []
    sec_count = 0
    max_secs = int(max_hours * 3600)

    for frame in reader:
        sec_buffer.append(frame["data"])
        if len(sec_buffer) == frames_per_sec:
            sec_tensor = torch.stack(sec_buffer).permute(0, 3, 1, 2).float() / 255.0
            sec_tensor = F.interpolate(sec_tensor, size=(384, 384), mode="bilinear", align_corners=False)
            sec_tensor = sec_tensor.unsqueeze(0).to(device)
            with torch.no_grad():
                tok = vid_enc.encode_second(sec_tensor)
            all_video_tokens.append(tok.cpu())
            sec_buffer = []
            sec_count += 1
            if sec_count % 300 == 0:
                print(f"  Encoded {sec_count//60}m {sec_count%60}s...")
            if sec_count >= max_secs:
                break

    if sec_buffer and sec_count < max_secs:
        while len(sec_buffer) < frames_per_sec:
            sec_buffer.append(sec_buffer[-1])
        sec_tensor = torch.stack(sec_buffer).permute(0, 3, 1, 2).float() / 255.0
        sec_tensor = F.interpolate(sec_tensor, size=(384, 384), mode="bilinear", align_corners=False)
        sec_tensor = sec_tensor.unsqueeze(0).to(device)
        with torch.no_grad():
            tok = vid_enc.encode_second(sec_tensor)
        all_video_tokens.append(tok.cpu())
        sec_count += 1

    video_tokens = torch.cat(all_video_tokens, dim=1).to(device)

    # Audio via torchaudio
    audio_tokens_list = []
    try:
        import torchaudio
        waveform, sr = torchaudio.load(mp4_path, normalize=True)
        if sr != 16000:
            waveform = T_audio.Resample(int(sr), 16000)(waveform)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)

        mel_transform = T_audio.MelSpectrogram(sample_rate=16000, n_fft=400, n_mels=128, hop_length=160)
        mel = mel_transform(waveform).squeeze(0).transpose(0, 1)

        sec_size = int(16000 / 160)
        for i in range(0, mel.size(0), sec_size):
            chunk = mel[i:i+sec_size].unsqueeze(0).to(device)
            if chunk.size(1) < sec_size:
                chunk = F.pad(chunk, (0, 0, 0, sec_size - chunk.size(1)))
            with torch.no_grad():
                tok = model.audio_encoder.encode_second(chunk)
            audio_tokens_list.append(tok.cpu())
    except Exception as e:
        print(f"  ⚠ Audio skipped: {e}")

    audio_tokens = torch.cat(audio_tokens_list, dim=1).to(device) if audio_tokens_list else None
    print(f"✅ Video tokens: {video_tokens.shape[1]} | Audio tokens: {audio_tokens.shape[1] if audio_tokens is not None else 0}")
    return video_tokens, audio_tokens


def main():
    parser = argparse.ArgumentParser(description="DeepSeek Dythos Video Inference")
    parser.add_argument("--variant", type=str, default="pro", choices=["pro", "flash"])
    parser.add_argument("--mp4", type=str, required=True)
    parser.add_argument("--prompt", type=str, required=True)
    parser.add_argument("--thinking-level", type=int, default=5)
    parser.add_argument("--max-tokens", type=int, default=1024)
    args = parser.parse_args()

    device = "cuda"
    cfg = DythosConfig(variant=args.variant)
    model = DeepSeekDythos(cfg).to(device=device, dtype=torch.bfloat16)
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/DeepSeek-V4-Pro", trust_remote_code=True)

    print("🔧 Encoding video (this may take a minute for 1hr content)...")
    video_tokens, audio_tokens = stream_encode_mp4(model, args.mp4, max_hours=1.0)

    engine = DythosInferenceEngine(model, tokenizer)
    result = engine.generate(
        [args.prompt],
        video_tokens=video_tokens,
        audio_tokens=audio_tokens,
        thinking_level=args.thinking_level,
        variant_idx=0 if args.variant == "pro" else 1,
        max_new_tokens=args.max_tokens,
    )
    print("\n🎬 Model Output:\n", result[0])


if __name__ == "__main__":
    main()
