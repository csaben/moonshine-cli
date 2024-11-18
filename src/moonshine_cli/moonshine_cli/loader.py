from pathlib import Path
from typing import Iterator, Optional, Union

import librosa
import torch


class AudioChunkLoader:
    def __init__(
        self,
        audio_path: Union[str, Path],
        chunk_duration: float = 60.0,
        sr: int = 22050,
        overlap: float = 0.0,
        max_duration: float = 60.0,  # Added maximum duration check
    ):
        """Initialize the audio chunk loader.

        Args:
            audio_path: Path to the audio file
            chunk_duration: Duration of each chunk in seconds
            sr: Sample rate to load the audio
            overlap: Overlap between chunks in seconds
            max_duration: Maximum allowed duration for a chunk in seconds
        """
        self.audio_path = Path(audio_path)
        # Ensure chunk duration doesn't exceed max_duration
        self.chunk_duration = min(chunk_duration, max_duration)
        self.sr = sr
        self.overlap = overlap

        # Load audio file and convert to torch tensor with expanded dimensions
        audio_np, self.actual_sr = librosa.load(self.audio_path, sr=self.sr, mono=True)
        self.audio = torch.from_numpy(audio_np).float().unsqueeze(0)

        # Calculate chunk size in samples (duration * sample_rate)
        self.chunk_size = int(self.chunk_duration * self.sr)
        self.overlap_size = int(self.overlap * self.sr)

        # Calculate total chunks
        self.total_samples = self.audio.shape[1]
        self.step_size = self.chunk_size - self.overlap_size
        self.total_chunks = max(
            1, (self.total_samples - self.overlap_size) // self.step_size
        )

        # Verify chunk duration
        chunk_duration_seconds = self.chunk_size / self.sr
        print(f"Actual chunk duration: {chunk_duration_seconds:.2f} seconds")

        if chunk_duration_seconds > max_duration:
            raise ValueError(
                f"Chunk duration ({chunk_duration_seconds:.2f}s) exceeds maximum allowed duration ({max_duration}s)"
            )

    def __len__(self) -> int:
        return self.total_chunks

    def __iter__(self) -> Iterator[torch.Tensor]:
        """Yield chunks of audio data as torch tensors."""
        for i in range(self.total_chunks):
            chunk = self.get_chunk(i)
            if chunk is not None:
                # Verify chunk duration
                chunk_duration = chunk.shape[1] / self.sr
                if chunk_duration <= 64.0:  # Moonshine's max duration
                    yield chunk

    def get_chunk(self, index: int) -> Optional[torch.Tensor]:
        """Get a specific chunk by index."""
        if index >= self.total_chunks or index < 0:
            return None

        start = index * self.step_size
        end = start + self.chunk_size

        if end > self.total_samples:
            chunk = torch.zeros((1, self.chunk_size), dtype=torch.float32)
            chunk[0, : self.total_samples - start] = self.audio[0, start:]
        else:
            chunk = self.audio[:, start:end]

        return chunk

    def get_chunk_duration(self, chunk: torch.Tensor) -> float:
        """Calculate duration of a chunk in seconds."""
        return chunk.shape[1] / self.sr
