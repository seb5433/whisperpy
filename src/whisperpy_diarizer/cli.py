import sys
import click
import whisper
import os
import json
from pathlib import Path
from typing import Dict, List, Tuple, Any
import torch
from pyannote.audio import Pipeline


def diar(args):
    return max(args, key=len)


def check_gpu_availability() -> Tuple[bool, str]:
    """Check if GPU is available and return status info."""
    if torch.cuda.is_available():
        device_name = torch.cuda.get_device_name()
        memory_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
        return True, f"CUDA: {device_name} ({memory_gb:.1f}GB)"
    elif torch.backends.mps.is_available():
        return True, "Apple Metal Performance Shaders (MPS)"
    else:
        return False, "No GPU acceleration available (CUDA or MPS)"


def get_hf_token() -> str:
    """Get HuggingFace token from environment variable."""
    token = os.getenv('HUGGINGFACE_TOKEN')
    if not token:
        raise click.ClickException(
            "HUGGINGFACE_TOKEN not found. Please set it using:\n"
            "export HUGGINGFACE_TOKEN=your_token_here\n"
            "Get your token from: https://huggingface.co/settings/tokens"
        )
    return token


def perform_diarization(audio_file: str, hf_token: str, verbose: bool = False, use_gpu: bool = False) -> Dict[str, Any]:
    """Perform speaker diarization using pyannote.audio."""
    if verbose:
        click.echo("Loading speaker diarization pipeline...")
        if use_gpu:
            if torch.cuda.is_available():
                click.echo("Using CUDA GPU acceleration for diarization...")
            elif torch.backends.mps.is_available():
                click.echo("⚠️  Apple Metal has compatibility issues with pyannote.audio, using CPU for diarization")
                click.echo("    (Whisper will still use GPU acceleration)")
            else:
                click.echo("GPU requested but not available, falling back to CPU...")
    
    # Initialize the speaker diarization pipeline
    pipeline = Pipeline.from_pretrained(
        "pyannote/speaker-diarization-3.1",
        use_auth_token=hf_token
    )
    
    # Only use CUDA for diarization, not MPS due to compatibility issues
    # MPS has known issues with sparse tensors used by pyannote.audio
    if use_gpu and torch.cuda.is_available():
        pipeline = pipeline.to(torch.device("cuda"))
        if verbose:
            click.echo("Successfully loaded diarization pipeline on CUDA GPU")
    elif verbose and use_gpu:
        click.echo("Keeping diarization pipeline on CPU for compatibility")
    
    # Run diarization
    if verbose:
        click.echo("Performing speaker diarization...")
    
    diarization = pipeline(audio_file)
    
    # Convert to a more usable format
    speakers = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speakers.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker,
            "duration": turn.end - turn.start
        })
    
    return {
        "speakers": speakers,
        "num_speakers": len({seg["speaker"] for seg in speakers})
    }


def match_segments_with_speakers(whisper_segments: List[Dict], speaker_segments: List[Dict]) -> List[Dict]:
    """Match Whisper transcription segments with speaker diarization segments."""
    matched_segments = []
    
    for whisper_seg in whisper_segments:
        whisper_start = whisper_seg["start"]
        whisper_end = whisper_seg["end"]
        whisper_mid = (whisper_start + whisper_end) / 2
        
        # Find the speaker segment that best overlaps with this whisper segment
        best_speaker = "Unknown"
        best_overlap = 0
        
        for speaker_seg in speaker_segments:
            # Calculate overlap
            overlap_start = max(whisper_start, speaker_seg["start"])
            overlap_end = min(whisper_end, speaker_seg["end"])
            overlap = max(0, overlap_end - overlap_start)
            
            if overlap > best_overlap:
                best_overlap = overlap
                best_speaker = speaker_seg["speaker"]
        
        # If no good overlap found, use the speaker active at the midpoint
        if best_overlap == 0:
            for speaker_seg in speaker_segments:
                if speaker_seg["start"] <= whisper_mid <= speaker_seg["end"]:
                    best_speaker = speaker_seg["speaker"]
                    break
        
        matched_segments.append({
            "start": whisper_start,
            "end": whisper_end,
            "duration": whisper_end - whisper_start,
            "text": whisper_seg["text"].strip(),
            "speaker": best_speaker
        })
    
    return matched_segments


def format_output_for_llm(segments: List[Dict], num_speakers: int) -> Dict[str, Any]:
    """Format the output in an LLM-friendly structure."""
    # Group segments by speaker
    speakers_text = {}
    conversation = []
    
    for segment in segments:
        speaker = segment["speaker"]
        if speaker not in speakers_text:
            speakers_text[speaker] = []
        
        speakers_text[speaker].append({
            "text": segment["text"],
            "start": round(segment["start"], 2),
            "end": round(segment["end"], 2),
            "duration": round(segment["duration"], 2)
        })
        
        conversation.append({
            "timestamp": f"{int(segment['start']//60):02d}:{int(segment['start']%60):02d}",
            "speaker": speaker,
            "text": segment["text"]
        })
    
    # Create summary
    total_duration = max(seg["end"] for seg in segments) if segments else 0
    
    return {
        "metadata": {
            "total_duration_seconds": round(total_duration, 2),
            "total_duration_formatted": f"{int(total_duration//60):02d}:{int(total_duration%60):02d}",
            "num_speakers": num_speakers,
            "num_segments": len(segments)
        },
        "speakers": {
            speaker: {
                "total_segments": len(texts),
                "segments": texts
            } for speaker, texts in speakers_text.items()
        },
        "conversation": conversation,
        "full_transcript": " ".join(seg["text"] for seg in segments)
    }


@click.group()
def cli():
    """
    WhisperPy: Audio transcription with speaker diarization.
    
    Use --boost flag for GPU acceleration (requires CUDA).
    Check GPU status with: whipy gpu-status
    """
    pass


@cli.command()
def gpu_status():
    """Check GPU availability and status for acceleration."""
    gpu_available, gpu_info = check_gpu_availability()
    
    if gpu_available:
        click.echo(f"✅ GPU acceleration available: {gpu_info}")
        click.echo("You can use --boost flag for faster processing")
        
        # Additional info for different GPU types
        if "CUDA" in gpu_info:
            click.echo("💡 NVIDIA CUDA GPU detected")
        elif "MPS" in gpu_info:
            click.echo("💡 Apple Silicon GPU detected (Metal Performance Shaders)")
            click.echo("Note: Whisper uses GPU, but diarization may fall back to CPU due to compatibility")
    else:
        click.echo(f"❌ GPU acceleration not available: {gpu_info}")
        click.echo("For GPU acceleration:")
        click.echo("  • NVIDIA GPUs: Install CUDA-compatible PyTorch")
        click.echo("  • Apple Silicon Macs: Install PyTorch with MPS support")
        click.echo("Visit: https://pytorch.org/get-started/locally/")


@cli.command()
@click.argument('token')
def set_token(token):
    """
    Set the HuggingFace token for accessing pyannote models.
    
    TOKEN: Your HuggingFace access token
    
    Get your token from: https://huggingface.co/settings/tokens
    """
    # Store token in a config file in user's home directory
    config_dir = Path.home() / '.config' / 'whisperpy'
    config_dir.mkdir(parents=True, exist_ok=True)
    config_file = config_dir / 'config.json'
    
    config = {}
    if config_file.exists():
        with open(config_file, 'r') as f:
            config = json.load(f)
    
    config['huggingface_token'] = token
    
    with open(config_file, 'w') as f:
        json.dump(config, f, indent=2)
    
    click.echo("HuggingFace token saved successfully!")
    click.echo("You can now use speaker diarization features.")


def get_stored_hf_token() -> str:
    """Get HuggingFace token from config file or environment."""
    # First try environment variable
    token = os.getenv('HUGGINGFACE_TOKEN')
    if token:
        return token
    
    # Then try config file
    config_file = Path.home() / '.config' / 'whisperpy' / 'config.json'
    if config_file.exists():
        try:
            with open(config_file, 'r') as f:
                config = json.load(f)
                token = config.get('huggingface_token')
                if token:
                    return token
        except (json.JSONDecodeError, KeyError):
            pass
    
    raise click.ClickException(
        "HUGGINGFACE_TOKEN not found. Please set it using:\n"
        "  whipy set-token YOUR_TOKEN\n"
        "or\n"
        "  export HUGGINGFACE_TOKEN=YOUR_TOKEN\n\n"
        "Get your token from: https://huggingface.co/settings/tokens"
    )


def write_json_output(output_path: Path, formatted_output: Dict[str, Any], verbose: bool):
    """Write JSON output to file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(formatted_output, f, indent=2, ensure_ascii=False)
    
    if verbose:
        click.echo(f"Diarized transcription saved to: {output_path}")
    else:
        click.echo(f"Transcription with speakers completed: {output_path}")


def print_conversation_summary(formatted_output: Dict[str, Any]):
    """Print conversation summary to stdout."""
    click.echo("\n=== TRANSCRIPTION SUMMARY ===")
    click.echo(f"Duration: {formatted_output['metadata']['total_duration_formatted']}")
    click.echo(f"Speakers detected: {formatted_output['metadata']['num_speakers']}")
    click.echo(f"Segments: {formatted_output['metadata']['num_segments']}")
    
    click.echo("\n=== CONVERSATION ===")
    for segment in formatted_output['conversation']:
        click.echo(f"[{segment['timestamp']}] {segment['speaker']}: {segment['text']}")


def write_txt_output(output_path: Path, formatted_output: Dict[str, Any]):
    """Write TXT output to file."""
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("=== TRANSCRIPTION SUMMARY ===\n")
        f.write(f"Duration: {formatted_output['metadata']['total_duration_formatted']}\n")
        f.write(f"Speakers detected: {formatted_output['metadata']['num_speakers']}\n")
        f.write(f"Segments: {formatted_output['metadata']['num_segments']}\n\n")
        
        f.write("=== CONVERSATION ===\n")
        for segment in formatted_output['conversation']:
            f.write(f"[{segment['timestamp']}] {segment['speaker']}: {segment['text']}\n")
    
    click.echo(f"Transcription completed: {output_path}")


def handle_diarization_output(formatted_output: Dict[str, Any], output: Path, format_type: str, verbose: bool):
    """Handle output for diarization results."""
    if format_type == 'json':
        write_json_output(output, formatted_output, verbose)
        print_conversation_summary(formatted_output)
    else:  # txt format
        write_txt_output(output, formatted_output)


def handle_simple_transcription(result: Dict, output: Path, verbose: bool):
    """Handle output for simple transcription without diarization."""
    with open(output, 'w', encoding='utf-8') as f:
        f.write(result['text'].strip())
    
    if verbose:
        click.echo(f"Transcription saved to: {output}")
    else:
        click.echo(f"Transcription completed: {output}")
        
    # Also print to stdout
    click.echo("\nTranscription:")
    click.echo(result['text'].strip())


@cli.command()
@click.argument('audio_file', type=click.Path(exists=True, readable=True))
@click.option('--model', '-m', default='base', 
              type=click.Choice(['tiny', 'base', 'small', 'medium', 'large', 'large-v2', 'large-v3', 'turbo']),
              help='Whisper model to use (default: base)')
@click.option('--output', '-o', type=click.Path(), 
              help='Output file path (default: same as input with .json extension)')
@click.option('--language', '-l', default=None,
              help='Language code (auto-detect if not specified)')
@click.option('--format', '-f', type=click.Choice(['json', 'txt']), default='json',
              help='Output format (default: json)')
@click.option('--boost', is_flag=True, help='Use GPU acceleration (CUDA/Apple Metal) for faster processing')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def transcribe(audio_file, model, output, language, format, boost, verbose):
    """
    Transcribe audio files with speaker diarization using OpenAI Whisper and pyannote.audio.
    
    AUDIO_FILE: Path to the audio file to transcribe and identify speakers
    """
    if verbose:
        click.echo(f"Loading Whisper model: {model}")
        if boost:
            if torch.cuda.is_available():
                click.echo("GPU acceleration enabled (CUDA)")
                click.echo(f"Using device: {torch.cuda.get_device_name()}")
            elif torch.backends.mps.is_available():
                click.echo("GPU acceleration enabled (Apple Metal)")
            else:
                click.echo("GPU acceleration requested but not available, using CPU")
    
    try:
        # Load and run Whisper with GPU support if requested
        device = "cpu"  # default
        if boost:
            if torch.cuda.is_available():
                device = "cuda"
            elif torch.backends.mps.is_available():
                device = "mps"
        
        whisper_model = whisper.load_model(model, device=device)
        
        if verbose:
            click.echo(f"Transcribing: {audio_file} (device: {device})")
        
        # Configure transcription options for GPU
        transcribe_options = {
            "language": language,
            "verbose": False
        }
        
        # Add GPU-specific optimizations if using CUDA
        if device == "cuda":
            transcribe_options["fp16"] = True  # Use half precision for faster GPU processing
        
        result = whisper_model.transcribe(audio_file, **transcribe_options)
        
        # Determine output file path
        if output is None:
            audio_path = Path(audio_file)
            extension = '.json' if format == 'json' else '.txt'
            output = audio_path.with_suffix(extension)
        
        # Get HuggingFace token for diarization
        try:
            hf_token = get_stored_hf_token()
        except click.ClickException as e:
            click.echo(str(e), err=True)
            sys.exit(1)
        
        # Perform diarization and transcription
        if verbose:
            click.echo("Performing speaker diarization...")
        
        diarization_result = perform_diarization(audio_file, hf_token, verbose, use_gpu=boost)
        matched_segments = match_segments_with_speakers(result['segments'], diarization_result['speakers'])
        formatted_output = format_output_for_llm(matched_segments, diarization_result['num_speakers'])
        
        # Save output
        if format == 'json':
            handle_diarization_output(formatted_output, Path(output), format, verbose)
        else:
            handle_simple_transcription(result, Path(output), verbose)
            
    except torch.cuda.OutOfMemoryError:
        click.echo("❌ GPU out of memory! Try:", err=True)
        click.echo("  1. Use a smaller model (e.g., --model base)", err=True)
        click.echo("  2. Run without --boost flag (CPU mode)", err=True)
        click.echo("  3. Close other GPU applications", err=True)
        sys.exit(1)
    except Exception as e:
        error_str = str(e)
        if ("MPS" in error_str or "mps" in error_str or "Metal" in error_str or 
              "SparseMPS" in error_str or "_sparse_coo_tensor" in error_str):
            click.echo("❌ Apple Metal GPU compatibility issue detected", err=True)
            click.echo("This is a known limitation with pyannote.audio on Apple Silicon", err=True)
            click.echo("💡 Whisper still benefits from GPU acceleration", err=True)
            click.echo("   Try running again - diarization will automatically use CPU", err=True)
        elif "CUDA" in error_str or "cuda" in error_str:
            click.echo(f"❌ CUDA GPU error: {e}", err=True)
            click.echo("Try running without --boost flag for CPU mode", err=True)
        else:
            click.echo(f"Error: {e}", err=True)
        sys.exit(1)


# Keep the old 'run' function for backward compatibility
run = transcribe


if __name__ == '__main__':
    cli()
