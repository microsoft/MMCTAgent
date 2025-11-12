import subprocess
import os
import math
import ffmpeg
import platform
from pathlib import Path
from loguru import logger


class VideoCompressor:
    def __init__(
        self,
        input_path,
        target_size_mb=500,
        audio_codec: str = "aac",
        preset: str = "fast",
        video_codec: str = "libx264",
        num_threads: int = 2,
        audio_bitrate_kbps=128,
        output_dir: str = "Compressed",

        # NEW: fast mode controls
        fast_mode: bool = True,                # use single-pass proxy-style compression
        max_width: int = 720,                  # max width for downscaling
        max_fps: float = 10.0,                 # cap FPS
        crf: int = 28,                         # CRF for x264 (lower=better quality)
        nvenc_cq: int = 28,                    # CQ for NVENC (lower=better quality)
    ):
        self.audio_codec = audio_codec
        self.preset = preset
        self.input_path = os.path.abspath(input_path)
        self.input_filename = os.path.basename(self.input_path)
        self.threads = str(num_threads)
        os.makedirs(output_dir, exist_ok=True)
        self.output_path = os.path.join(output_dir, self.input_filename)
        self.temp_null = "NUL" if os.name == "nt" else "/dev/null"
        self.target_size_kbits = target_size_mb * 1024 * 8
        self.audio_bitrate = audio_bitrate_kbps
        self.logger = logger

        self.fast_mode = fast_mode
        self.max_width = max_width
        self.max_fps = max_fps
        self.crf = crf
        self.nvenc_cq = nvenc_cq

        self.video_codec, self.use_gpu = self._detect_gpu_codec(video_codec)
        self.duration = self._get_duration()
        self.video_bitrate = self._calculate_video_bitrate()
        print(self.input_path, self.output_path)

    # ------------------------------------------------------------------
    # GPU detection utilities
    # ------------------------------------------------------------------
    def _detect_gpu_codec(self, preferred_codec):
        """Detect available GPU encoders and return appropriate codec"""
        gpu_codecs = {
            "nvidia": ["h264_nvenc", "hevc_nvenc"],
            "amd": ["h264_amf", "hevc_amf"],
            "intel": ["h264_qsv", "hevc_qsv"],
        }

        for codec in gpu_codecs["nvidia"]:
            if self._test_encoder(codec):
                self.logger.info(f"Using NVIDIA GPU encoder: {codec}")
                return codec, True

        for codec in gpu_codecs["amd"]:
            if self._test_encoder(codec):
                self.logger.info(f"Using AMD GPU encoder: {codec}")
                return codec, True

        for codec in gpu_codecs["intel"]:
            if self._test_encoder(codec):
                self.logger.info(f"Using Intel GPU encoder: {codec}")
                return codec, True

        self.logger.info(f"No GPU encoder available, using CPU encoder: {preferred_codec}")
        return preferred_codec, False

    def _test_encoder(self, encoder):
        """Test if a specific encoder is available"""
        try:
            result = subprocess.run(
                [
                    "ffmpeg",
                    "-hide_banner",
                    "-f",
                    "lavfi",
                    "-i",
                    "testsrc=duration=1:size=320x240:rate=1",
                    "-c:v",
                    encoder,
                    "-t",
                    "1",
                    "-f",
                    "null",
                    self.temp_null,
                ],
                capture_output=True,
                timeout=10,
            )
            return result.returncode == 0
        except (subprocess.TimeoutExpired, subprocess.CalledProcessError, FileNotFoundError):
            return False

    # ------------------------------------------------------------------
    # Metadata helpers
    # ------------------------------------------------------------------
    def _get_duration(self):
        try:
            probe = ffmpeg.probe(self.input_path)
            duration = float(probe["format"]["duration"])
            self.logger.info(f"Video duration: {duration:.2f} seconds")
            return duration
        except Exception as e:
            self.logger.exception("Could not get video duration")
            raise RuntimeError(f"Could not get video duration: {e}")

    def _calculate_video_bitrate(self):
        bitrate = max(
            1000,
            math.floor(
                (self.target_size_kbits - (self.audio_bitrate * self.duration)) / self.duration
            ),
        )
        self.logger.info(f"Target video bitrate: {bitrate} kbps")
        return bitrate

    async def _run_and_log(self, command, description):
        import asyncio
        
        self.logger.info(f"Starting: {description} of {self.input_path}")
        process = await asyncio.create_subprocess_exec(
            *command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        out, err = await process.communicate()
        self.logger.debug(f"--- {description} stderr ---\n{err.decode().strip()}")
        if process.returncode != 0:
            self.logger.error(f"{description} failed. See log for details.")
        else:
            self.logger.info(f"{description} completed successfully.")

    def _cleanup_temp_files(self):
        temp_files = ["ffmpeg2pass-0.log", "ffmpeg2pass-0.log.mbtree"]
        for f in temp_files:
            if os.path.exists(f):
                try:
                    os.remove(f)
                    self.logger.info(f"Deleted temp file: {f}")
                except Exception as e:
                    self.logger.warning(f"Could not delete temp file {f}: {e}")

    # ------------------------------------------------------------------
    # Fast single-pass compression
    # ------------------------------------------------------------------
    async def _compress_fast(self):
        """
        Fast proxy-style compression:
        - single pass
        - downscale spatially and temporally
        - CRF or CQ quality-based control
        - hardware accelerated if available
        """
        base_cmd = [
            "ffmpeg",
            "-y",
            "-hwaccel",
            "auto",
            "-i",
            self.input_path,
            "-r",
            str(self.max_fps),
            "-vf",
            f"scale='min({self.max_width},iw)':-2",
        ]

        if self.use_gpu and "nvenc" in self.video_codec:
            video_opts = [
                "-c:v",
                self.video_codec,
                "-preset",
                "fast",
                "-rc",
                "vbr",
                "-cq",
                str(self.nvenc_cq),
                "-b:v",
                "0",
            ]
        elif self.use_gpu and ("amf" in self.video_codec or "qsv" in self.video_codec):
            video_opts = [
                "-c:v",
                self.video_codec,
                "-preset",
                "fast",
                "-b:v",
                f"{self.video_bitrate}k",
            ]
        else:
            video_opts = [
                "-c:v",
                "libx264",
                "-preset",
                self.preset,
                "-crf",
                str(self.crf),
                "-tune",
                "fastdecode",
                "-threads",
                self.threads,
            ]

        audio_opts = ["-c:a", self.audio_codec, "-b:a", f"{self.audio_bitrate}k"]

        cmd = base_cmd + video_opts + audio_opts + [self.output_path]
        await self._run_and_log(cmd, "Fast Compression (single pass)")

    # ------------------------------------------------------------------
    # Original two-pass compression
    # ------------------------------------------------------------------
    async def _compress_twopass(self):
        """Two-pass size-targeted compression (slower but more precise)."""
        base_cmd = ["ffmpeg", "-y", "-i", str(self.input_path)]

        if self.use_gpu:
            if "nvenc" in self.video_codec:
                base_cmd.extend(["-hwaccel", "cuda"])
            elif "amf" in self.video_codec:
                base_cmd.extend(["-hwaccel", "dxva2"])
            elif "qsv" in self.video_codec:
                base_cmd.extend(["-hwaccel", "qsv"])

        if self.use_gpu and "nvenc" in self.video_codec:
            preset_option, preset_value = "-preset", "fast"
        elif self.use_gpu and ("amf" in self.video_codec or "qsv" in self.video_codec):
            preset_option, preset_value = "-preset", "fast"
        else:
            preset_option, preset_value = "-preset", self.preset

        cmd1 = base_cmd + [
            preset_option,
            preset_value,
            "-c:v",
            self.video_codec,
            "-b:v",
            f"{self.video_bitrate}k",
            "-pass",
            "1",
            "-an",
            "-f",
            "mp4",
            self.temp_null,
        ]
        if not self.use_gpu:
            cmd1.extend(["-threads", self.threads])

        cmd2 = base_cmd + [
            preset_option,
            preset_value,
            "-c:v",
            self.video_codec,
            "-b:v",
            f"{self.video_bitrate}k",
            "-pass",
            "2",
            "-map",
            "0:v",
            "-map",
            "0:a?",
            "-c:a",
            self.audio_codec,
            "-b:a",
            f"{self.audio_bitrate}k",
            str(self.output_path),
        ]
        if not self.use_gpu:
            cmd2.extend(["-threads", self.threads])

        await self._run_and_log(cmd1, "First Pass")
        await self._run_and_log(cmd2, "Second Pass")
        self._cleanup_temp_files()

    # ------------------------------------------------------------------
    # Main entrypoint
    # ------------------------------------------------------------------
    async def compress(self):
        """
        fast_mode=True  -> single-pass proxy compression (recommended for analytics)
        fast_mode=False -> 2-pass size-constrained compression
        """
        if self.fast_mode:
            await self._compress_fast()
        else:
            await self._compress_twopass()

        # Validate output
        try:
            probe = ffmpeg.probe(self.output_path)
            has_video = any(s["codec_type"] == "video" for s in probe["streams"])
            has_audio = any(s["codec_type"] == "audio" for s in probe["streams"])
            if not has_video:
                raise RuntimeError("Compressed video has no video stream!")
            if not has_audio:
                self.logger.warning("Compressed video has no audio stream (may not exist originally)")
            else:
                self.logger.info("Compressed video has both video and audio streams")
        except Exception as e:
            self.logger.warning(f"Could not verify compressed video streams: {e}")

        self.logger.info(f"Compression complete. Output saved to: {self.output_path}")


# ----------------------------------------------------------------------
# CLI runner
# ----------------------------------------------------------------------
if __name__ == "__main__":
    import argparse
    import asyncio

    parser = argparse.ArgumentParser(description="Compress video files to target size or proxy")
    parser.add_argument("input_path", help="Path to input video file")
    parser.add_argument("-s", "--size", type=int, default=500, help="Target size in MB (default: 500)")
    parser.add_argument("-o", "--output", default="Compressed", help="Output directory (default: Compressed)")
    parser.add_argument("--video-codec", default="libx264", help="Video codec (default: libx264)")
    parser.add_argument("--audio-codec", default="aac", help="Audio codec (default: aac)")
    parser.add_argument("--preset", default="fast", help="Encoding preset (default: fast)")
    parser.add_argument("--threads", type=int, default=2, help="Number of threads (default: 2)")
    parser.add_argument("--audio-bitrate", type=int, default=128, help="Audio bitrate in kbps (default: 128)")
    parser.add_argument("--fast-mode", action="store_true", default=False, help="Use fast single-pass proxy compression")
    parser.add_argument("--max-width", type=int, default=720, help="Max width in fast mode (default: 720)")
    parser.add_argument("--max-fps", type=float, default=10.0, help="Max fps in fast mode (default: 10)")
    parser.add_argument("--crf", type=int, default=28, help="CRF for x264 in fast mode (default: 28)")
    parser.add_argument("--nvenc-cq", type=int, default=28, help="CQ for NVENC in fast mode (default: 28)")

    args = parser.parse_args()

    compressor = VideoCompressor(
        input_path=args.input_path,
        target_size_mb=args.size,
        audio_codec=args.audio_codec,
        preset=args.preset,
        video_codec=args.video_codec,
        num_threads=args.threads,
        audio_bitrate_kbps=args.audio_bitrate,
        output_dir=args.output,
        fast_mode=args.fast_mode,
        max_width=args.max_width,
        max_fps=args.max_fps,
        crf=args.crf,
        nvenc_cq=args.nvenc_cq,
    )
    asyncio.run(compressor.compress())
