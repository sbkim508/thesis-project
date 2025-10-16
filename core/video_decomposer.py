"""Utilities for decomposing video into modality specific assets."""
from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

try:
    import cv2
except ImportError:  # pragma: no cover - optional dependency
    cv2 = None  # type: ignore

try:
    from moviepy.editor import AudioFileClip, VideoFileClip
except ImportError:  # pragma: no cover - optional dependency
    AudioFileClip = None  # type: ignore
    VideoFileClip = None  # type: ignore

LOGGER = logging.getLogger(__name__)


@dataclass
class VideoDecomposer:
    """Split a video into frames, audio, and optionally transcripts."""

    output_dir: Path
    fps: Optional[float] = None
    max_frames: Optional[int] = None

    def __post_init__(self) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def extract_frames(self, video_path: Path) -> List[Path]:
        """Extract frames from the video and return their file paths."""
        if cv2 is None:
            raise ImportError("opencv-python is required for frame extraction")
        capture = cv2.VideoCapture(str(video_path))
        if not capture.isOpened():
            raise RuntimeError(f"Unable to open video: {video_path}")
        frame_paths: List[Path] = []
        native_fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        target_fps = self.fps or native_fps or 1.0
        frame_interval = max(int(round(native_fps / target_fps)) if native_fps and target_fps else 1, 1)
        frame_index = 0
        saved_frames = 0
        while True:
            success, frame = capture.read()
            if not success:
                break
            if frame_index % frame_interval == 0:
                frame_file = self.output_dir / f"frame_{saved_frames:06d}.png"
                cv2.imwrite(str(frame_file), frame)
                frame_paths.append(frame_file)
                saved_frames += 1
                if self.max_frames and saved_frames >= self.max_frames:
                    break
            frame_index += 1
        capture.release()
        LOGGER.info("Extracted %d frames from %s", len(frame_paths), video_path)
        return frame_paths

    def extract_audio(self, video_path: Path) -> Path:
        """Extract audio track from the video and save it as a WAV file."""
        if VideoFileClip is None:
            raise ImportError("moviepy is required for audio extraction")
        video_clip = VideoFileClip(str(video_path))
        audio_path = self.output_dir / "audio.wav"
        if video_clip.audio is None:
            raise RuntimeError(f"Video {video_path} does not contain an audio track")
        video_clip.audio.write_audiofile(str(audio_path), logger=None)
        video_clip.close()
        LOGGER.info("Extracted audio to %s", audio_path)
        return audio_path

    def chunk_audio(self, audio_path: Path, duration: float = 30.0) -> List[Path]:
        """Split audio into fixed duration segments to feed downstream models."""
        if AudioFileClip is None:
            raise ImportError("moviepy is required for audio chunking")
        audio_clip = AudioFileClip(str(audio_path))
        chunks: List[Path] = []
        start = 0.0
        index = 0
        while start < audio_clip.duration:
            end = min(start + duration, audio_clip.duration)
            chunk_path = self.output_dir / f"audio_{index:03d}.wav"
            audio_clip.subclip(start, end).write_audiofile(str(chunk_path), logger=None)
            chunks.append(chunk_path)
            start = end
            index += 1
        audio_clip.close()
        LOGGER.info("Chunked audio into %d segments", len(chunks))
        return chunks


__all__ = ["VideoDecomposer"]
