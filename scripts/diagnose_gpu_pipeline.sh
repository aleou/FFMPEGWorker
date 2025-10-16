#!/bin/bash
# GPU Pipeline Diagnostic Script
# Run this inside the RunPod container to diagnose GPU pipeline issues

set -e

echo "======================================"
echo "🔍 GPU Pipeline Diagnostic Tool"
echo "======================================"
echo ""

# 1. Check NVIDIA GPU
echo "📊 1. Checking NVIDIA GPU..."
if command -v nvidia-smi &> /dev/null; then
    nvidia-smi --query-gpu=name,memory.total,driver_version,cuda_version --format=csv,noheader
    echo "✅ NVIDIA GPU detected"
else
    echo "❌ nvidia-smi not found!"
    exit 1
fi
echo ""

# 2. Check Python/PyTorch CUDA
echo "🐍 2. Checking PyTorch CUDA..."
python3 - <<'PYTHON'
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"GPU name: {torch.cuda.get_device_name(0)}")
    props = torch.cuda.get_device_properties(0)
    vram_gb = props.total_memory / (1024**3)
    print(f"VRAM: {vram_gb:.1f} GB")
    print(f"Compute capability: {props.major}.{props.minor}")
    print("✅ PyTorch CUDA working")
else:
    print("❌ PyTorch CUDA not available!")
PYTHON
echo ""

# 3. Check FFmpeg
echo "🎬 3. Checking FFmpeg..."
if command -v ffmpeg &> /dev/null; then
    ffmpeg -version | head -n 1
    echo "✅ FFmpeg installed"
else
    echo "❌ FFmpeg not found!"
    exit 1
fi
echo ""

# 4. Check FFmpeg hardware acceleration
echo "🚀 4. Checking FFmpeg hardware acceleration support..."
echo "Available hardware accelerators:"
ffmpeg -hwaccels 2>&1 | grep -v "^ffmpeg" | grep -v "^Hardware"
if ffmpeg -hwaccels 2>&1 | grep -q cuda; then
    echo "✅ CUDA hardware decode available"
else
    echo "⚠️  CUDA hardware decode NOT available"
fi
echo ""

# 5. Check NVENC support
echo "🎥 5. Checking NVENC hardware encoding support..."
if ffmpeg -codecs 2>&1 | grep -q h264_nvenc; then
    echo "✅ h264_nvenc available"
    ffmpeg -hide_banner -h encoder=h264_nvenc 2>&1 | grep -E "(Supported pixel|presets:)" | head -n 3
else
    echo "❌ h264_nvenc NOT available!"
fi
echo ""

# 6. Test NVENC encoding
echo "🧪 6. Testing NVENC encoding..."
TEST_FILE="/tmp/test_nvenc.mp4"
if ffmpeg -y -f lavfi -i testsrc=duration=1:size=640x480:rate=30 -frames:v 30 \
    -c:v h264_nvenc -preset p7 -tune hq -rc vbr -cq 19 "$TEST_FILE" 2>&1 | tail -n 5; then
    if [ -f "$TEST_FILE" ]; then
        SIZE=$(stat -f%z "$TEST_FILE" 2>/dev/null || stat -c%s "$TEST_FILE" 2>/dev/null)
        echo "✅ NVENC test successful! Output: $SIZE bytes"
        rm "$TEST_FILE"
    fi
else
    echo "❌ NVENC test failed!"
fi
echo ""

# 7. Check PyAV
echo "📦 7. Checking PyAV (av)..."
python3 - <<'PYTHON'
import av
print(f"PyAV version: {av.__version__}")
print(f"FFmpeg version: {av.codec.codecs_available}")
# Try to list available codecs
codecs = []
try:
    for codec_name in ['h264_nvenc', 'hevc_nvenc', 'libx264']:
        if codec_name in av.codec.codecs_available:
            codecs.append(codec_name)
except:
    pass
if codecs:
    print(f"Available codecs: {', '.join(codecs)}")
print("✅ PyAV imported successfully")
PYTHON
echo ""

# 8. Check GPU memory
echo "💾 8. Checking GPU memory usage..."
nvidia-smi --query-gpu=memory.used,memory.free,memory.total --format=csv,noheader,nounits
echo ""

# 9. Test PyAV CUDA decode
echo "🔬 9. Testing PyAV CUDA hardware decode..."
python3 - <<'PYTHON'
import av
import sys

# Create test video first
import subprocess
subprocess.run([
    "ffmpeg", "-y", "-f", "lavfi", "-i", "testsrc=duration=1:size=640x480:rate=30",
    "-c:v", "libx264", "-preset", "ultrafast", "-frames:v", "30", "/tmp/test_input.mp4"
], capture_output=True)

try:
    # Try CUDA decode
    container = av.open("/tmp/test_input.mp4", options={"hwaccel": "cuda"})
    video_stream = container.streams.video[0]
    frame_count = 0
    for frame in container.decode(video_stream):
        frame_count += 1
        if frame_count >= 5:
            break
    container.close()
    print(f"✅ PyAV CUDA hardware decode successful! Decoded {frame_count} frames")
except av.AVError as e:
    print(f"⚠️  PyAV CUDA decode failed: {e}")
    print("   Falling back to software decode...")
    try:
        container = av.open("/tmp/test_input.mp4")
        video_stream = container.streams.video[0]
        frame_count = 0
        for frame in container.decode(video_stream):
            frame_count += 1
            if frame_count >= 5:
                break
        container.close()
        print(f"✅ PyAV software decode successful! Decoded {frame_count} frames")
    except Exception as e2:
        print(f"❌ PyAV decode completely failed: {e2}")
        sys.exit(1)
except Exception as e:
    print(f"❌ PyAV test failed: {e}")
    sys.exit(1)
finally:
    import os
    try:
        os.remove("/tmp/test_input.mp4")
    except:
        pass
PYTHON
echo ""

echo "======================================"
echo "✅ Diagnostic complete!"
echo "======================================"
echo ""
echo "Summary:"
echo "- If all checks passed, GPU pipeline should work"
echo "- If NVENC failed, GPU pipeline will use CPU encoding (slower but still much faster)"
echo "- If PyAV CUDA decode failed, it will fallback to software decode (minimal impact)"
echo ""
echo "Next step: Run a watermark removal job and check logs for:"
echo "  'GPU pipeline: Starting video decode...'"
echo "  'GPU pipeline: Decoded X segments'"
echo "  'GPU pipeline: Converted X frames successfully'"
echo "  'NVENC encoding successful' or 'CPU encoding successful'"
