import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from obspy import read
from scipy.signal import hilbert, find_peaks
from scipy.stats import kurtosis
from datetime import datetime, timedelta
import pytz
from tqdm import tqdm
import multiprocessing

# === CONFIGURATION ===
SAMPLING_RATE = 51.2  # Default, but auto-detect per file if possible
MIN_EVENT_DURATION_SEC = 0.2  # Minimum event duration (infra20 quantization ~0.02s)
MAX_GAP_SEC = 1.0  # Merge segments closer than this
CLUSTER_PROXIMITY_SEC = 5.0  # For cluster assignment
SPECTROGRAM_NFFT = 256  # You may adjust for visible resolution
MECH_HARMONIC_MIN = 2  # How many harmonics for mechanical label
SNR_REF_SEC = 2  # How much background to sample before/after event for SNR
ARTIFACTS_SUFFIX = "_artifacts"
PACIFIC_TZ = pytz.timezone('US/Pacific')

# Mechanical features: Periodic (harmonics), long duration, freq drift ok
# Impulse: High kurtosis, short, broadband
# Attack: SNR, freq prominence, low freq std, not mechanical/impulse

def ensure_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

def compute_snr(event_data, bg1, bg2):
    event_rms = np.sqrt(np.mean(np.square(event_data)))
    bg_data = np.concatenate([bg1, bg2]) if len(bg1)+len(bg2) > 0 else np.array([1e-6])
    bg_rms = np.sqrt(np.mean(np.square(bg_data)))
    if bg_rms == 0: bg_rms = 1e-6
    snr_db = 20 * np.log10(event_rms / bg_rms)
    return snr_db

def harmonic_count(freqs, amps, prominence_ratio=0.2):
    # Find fundamental, count peaks at near-integer multiples above threshold
    peaks, prop = find_peaks(amps, height=np.max(amps)*prominence_ratio)
    harmonics = 0
    if len(peaks) < 1:
        return 0
    base_idx = peaks[np.argmax(amps[peaks])]
    base_freq = freqs[base_idx]
    for n in range(2, 8):
        harmonic_freq = base_freq * n
        idx = np.argmin(np.abs(freqs - harmonic_freq))
        if amps[idx] > np.max(amps)*prominence_ratio:
            harmonics += 1
    return harmonics

def classify_waveform(seg, fs):
    analytic = hilbert(seg)
    envelope = np.abs(analytic)
    kurt = kurtosis(seg)
    if kurt > 8 and len(seg) < fs:
        return 'impulse'
    # Pulse train: regular, repeating peaks
    peaks, _ = find_peaks(envelope, distance=fs*0.05)
    if len(peaks) >= 3:
        intervals = np.diff(peaks) / fs
        if np.std(intervals) < 0.15 * np.mean(intervals):
            return 'pulse train'
    # AM: envelope has low-freq component modulating carrier
    env_fft = np.abs(np.fft.rfft(envelope * np.hanning(len(envelope))))
    env_freqs = np.fft.rfftfreq(len(envelope), d=1/fs)
    dom_env_idx = np.argmax(env_fft[1:]) + 1
    dom_env_freq = env_freqs[dom_env_idx]
    if 0.5 < dom_env_freq < 10 and np.max(env_fft) > 3*np.mean(env_fft):
        return 'AM'
    # Sine: narrowband, low std, single freq
    return 'sine'  # fallback; could add 'broadband' with further logic

def classify_event(seg, fs, dom_freq, freq_std, amps, freqs, harmonic_n, kurt, snr_db):
    # Probabilities are illustrative; refine as needed
    prob_mechanical = 0.0
    prob_impulse = 0.0
    prob_attack = 0.0

    # Impulse: high kurtosis, short, broadband
    if kurt > 8 and len(seg) < 0.5*fs:
        prob_impulse = min(1.0, (kurt-8)/8 + (0.5 - len(seg)/fs))
    # Mechanical: harmonics, moderate kurt, long duration, low SNR ok
    if harmonic_n >= MECH_HARMONIC_MIN:
        prob_mechanical = min(1.0, 0.4 + 0.1*harmonic_n + 0.5*(freq_std/5.0))
    # Attack: SNR>8dB, dom freq amp > 2x mean, std < 0.3Hz, not mechanical/impulse
    dom_idx = np.argmin(np.abs(freqs-dom_freq))
    dom_amp = amps[dom_idx]
    mean_amp = np.mean(amps)
    if snr_db > 8 and dom_amp > 2*mean_amp and freq_std < 0.3 and prob_mechanical < 0.5 and prob_impulse < 0.5:
        prob_attack = min(1.0, 0.5 + 0.5*((snr_db-8)/10.0 + (dom_amp/mean_amp-2)/2 + (0.3-freq_std)/0.3))
    # Clamp to [0,1]
    prob_mechanical = min(max(prob_mechanical,0.0),1.0)
    prob_impulse = min(max(prob_impulse,0.0),1.0)
    prob_attack = min(max(prob_attack,0.0),1.0)
    # Ambiguous if all low
    return prob_mechanical, prob_impulse, prob_attack

def segment_events(data, sampling_rate, min_duration_sec, min_ampl=0.08, max_gap_sec=MAX_GAP_SEC):
    analytic = hilbert(data)
    envelope = np.abs(analytic)
    above = envelope > min_ampl
    events = []
    in_event = False
    start = None
    last_idx = None
    for i, v in enumerate(above):
        if v and not in_event:
            in_event = True
            start = i
        if not v and in_event:
            end = i
            dur = (end - start) / sampling_rate
            if dur >= min_duration_sec:
                events.append((start, end))
            in_event = False
            start = None
        last_idx = i
    if in_event and start is not None:
        end = last_idx + 1
        dur = (end - start) / sampling_rate
        if dur >= min_duration_sec:
            events.append((start, end))
    # Merge if gap < max_gap
    merged = []
    if events:
        prev = list(events[0])
        for s, e in events[1:]:
            gap = (s - prev[1]) / sampling_rate
            if gap < max_gap_sec:
                prev[1] = e
            else:
                merged.append(tuple(prev))
                prev = [s, e]
        merged.append(tuple(prev))
    return merged

def get_cluster_ids(event_starts, proximity_sec, fs):
    clusters = []
    cluster_id = 1
    prev_end = None
    for i, start in enumerate(event_starts):
        if prev_end is None or (start - prev_end) > int(proximity_sec * fs):
            cluster_id += 1
        clusters.append(cluster_id)
        prev_end = start
    return clusters

def process_file(filepath, month, day, hour, artifact_dir):
    st = read(filepath)
    tr = st[0]
    data = tr.data
    sampling_rate = tr.stats.sampling_rate if hasattr(tr.stats, "sampling_rate") else SAMPLING_RATE

    events = segment_events(data, sampling_rate, MIN_EVENT_DURATION_SEC)
    event_rows = []
    cluster_ids = get_cluster_ids([s for (s, e) in events], CLUSTER_PROXIMITY_SEC, sampling_rate)

    file_datetime = datetime(2025, int(month), int(day), int(hour), tzinfo=pytz.UTC)
    for idx, ((start, end), cluster) in enumerate(zip(events, cluster_ids)):
        seg = data[start:end]
        duration = (end - start) / sampling_rate
        if duration < MIN_EVENT_DURATION_SEC:
            continue
        # FFT
        n = len(seg)
        windowed = seg * np.hanning(n)
        freqs = np.fft.rfftfreq(n, d=1/sampling_rate)
        amps = np.abs(np.fft.rfft(windowed))
        dom_idx = np.argmax(amps)
        dom_freq = freqs[dom_idx]
        # Only allow events within instrument bandwidth (<20 Hz)
        if dom_freq < 2.0 or dom_freq > 20.0:
            continue
        # Remove fixed “2.98 Hz” bin events (can be parameterized)
        if 2.97 < dom_freq < 2.99:
            continue
        # Energy prominence/ambiguity check
        sorted_amps = np.sort(amps)[::-1]
        dom_amp = amps[dom_idx]
        next_strongest = sorted_amps[1] if len(sorted_amps) > 1 else 1e-6
        prominence = dom_amp / (np.mean(amps)+1e-6)
        if dom_amp < 1.5*next_strongest:
            event_type = 'ambiguous'
        else:
            event_type = classify_waveform(seg, sampling_rate)
        freq_std = np.std(freqs[amps > 0.5*dom_amp]) if np.any(amps > 0.5*dom_amp) else 0.0
        harmonic_n = harmonic_count(freqs, amps)
        kurt = kurtosis(seg)
        # SNR: take 2s before/after (or as much as available), avoiding event
        pre_start = max(0, start - int(SNR_REF_SEC*sampling_rate))
        post_end = min(len(data), end + int(SNR_REF_SEC*sampling_rate))
        bg1 = data[pre_start:start] if start - pre_start > 5 else np.array([])
        bg2 = data[end:post_end] if post_end - end > 5 else np.array([])
        snr_db = compute_snr(seg, bg1, bg2)
        # Event feature calculations
        spi_peak = np.max(np.abs(seg))         # In counts
        spi_mean = np.mean(np.abs(seg))        # In counts
        spi_peak_pa = spi_peak * 0.001         # Convert to Pascals
        spi_mean_pa = spi_mean * 0.001         # Convert to Pascals
        amp_peak_db = 20 * np.log10(spi_peak_pa / 0.00002) if spi_peak_pa > 0 else 0
        amp_mean_db = 20 * np.log10(spi_mean_pa / 0.00002) if spi_mean_pa > 0 else 0
        # Probabilities
        prob_mechanical, prob_impulse, prob_attack = classify_event(
            seg, sampling_rate, dom_freq, freq_std, amps, freqs, harmonic_n, kurt, snr_db
        )
        # Times
        event_start_time_utc = file_datetime + timedelta(seconds=start / sampling_rate)
        event_end_time_utc = file_datetime + timedelta(seconds=end / sampling_rate)
        event_start_time_local = event_start_time_utc.astimezone(PACIFIC_TZ)
        event_end_time_local = event_end_time_utc.astimezone(PACIFIC_TZ)
        # Spectrogram artifact
        spec_fname = f"event_{month}_{day}_{hour}_{idx+1}.png"
        spec_path = os.path.join(artifact_dir, spec_fname)
        plt.figure(figsize=(6,3))
        plt.specgram(seg, Fs=sampling_rate, NFFT=SPECTROGRAM_NFFT, noverlap=SPECTROGRAM_NFFT//2)
        plt.title(f"Event {idx+1}: {event_type}, {dom_freq:.2f} Hz, {duration:.2f}s")
        plt.xlabel("Time (s)")
        plt.ylabel("Frequency (Hz)")
        plt.colorbar(label='dB')
        plt.tight_layout()
        plt.savefig(spec_path)
        plt.close()
        # Row
        event_rows.append({
            "Event ID": idx+1,
            "Cluster ID": cluster,
            "Start Time (UTC)": event_start_time_utc.strftime('%Y-%m-%d %H:%M:%S'),
            "Start Time (Local Pacific)": event_start_time_local.strftime('%Y-%m-%d %H:%M:%S'),
            "End Time (UTC)": event_end_time_utc.strftime('%Y-%m-%d %H:%M:%S'),
            "End Time (Local Pacific)": event_end_time_local.strftime('%Y-%m-%d %H:%M:%S'),
            "Duration (s)": duration,
            "Dominant Frequency (Hz)": dom_freq,
            "Frequency Stddev (Hz)": freq_std,
            "SPI (peak, counts)": spi_peak,
            "SPI (mean, counts)": spi_mean,
            "SPI (peak, Pa)": spi_peak_pa,
            "SPI (mean, Pa)": spi_mean_pa,
            "Amplitude (peak dB SPL)": amp_peak_db,
            "Amplitude (mean dB SPL)": amp_mean_db,
            "Harmonic Count": harmonic_n,
            "Kurtosis": kurt,
            "SNR (dB)": snr_db,
            "Waveform Type": event_type,
            "Mechanical Probability": prob_mechanical,
            "Impulse Probability": prob_impulse,
            "Attack Probability": prob_attack,
            "Spectrogram File": os.path.relpath(spec_path),
        })
    return event_rows

def process_all_files(input_dir, artifact_dir):
    rows = []
    months = sorted([m for m in os.listdir(input_dir) if os.path.isdir(os.path.join(input_dir, m))])
    total_files = sum(
        1 for m in months for d in os.listdir(os.path.join(input_dir, m))
        if os.path.isdir(os.path.join(input_dir, m, d))
        for f in os.listdir(os.path.join(input_dir, m, d)) if f.endswith('.sac')
    )
    pbar = tqdm(total=total_files, desc="Processing SAC files")
    for month in months:
        month_dir = os.path.join(input_dir, month)
        days = sorted([d for d in os.listdir(month_dir) if os.path.isdir(os.path.join(month_dir, d))])
        for day in days:
            day_dir = os.path.join(month_dir, day)
            for hourfile in sorted(os.listdir(day_dir)):
                if not hourfile.endswith('.sac'): continue
                hour = hourfile.split('.')[0]
                filepath = os.path.join(day_dir, hourfile)
                try:
                    file_rows = process_file(filepath, month, day, hour, artifact_dir)
                    rows.extend(file_rows)
                except Exception as e:
                    print(f"Failed to process {filepath}: {e}")
                pbar.update(1)
    pbar.close()
    return rows

def plot_histograms(df, artifact_dir):
    # Event type count
    plt.figure(figsize=(8,5))
    df['Waveform Type'].value_counts().plot(kind='bar')
    plt.title("Event Waveform Type Histogram")
    plt.xlabel("Waveform Type")
    plt.ylabel("Event Count")
    plt.tight_layout()
    hist1_path = os.path.join(artifact_dir, "event_waveform_histogram.png")
    plt.savefig(hist1_path)
    plt.close()
    # Probability density histograms
    for col in ["Mechanical Probability", "Impulse Probability", "Attack Probability"]:
        plt.figure(figsize=(8,5))
        df[col].hist(bins=20)
        plt.title(f"{col} Histogram")
        plt.xlabel(col)
        plt.ylabel("Event Count")
        plt.tight_layout()
        plt.savefig(os.path.join(artifact_dir, f"{col.replace(' ','_').lower()}_histogram.png"))
        plt.close()

def main(input_dir, output_csv):
    artifact_dir = os.path.splitext(output_csv)[0] + ARTIFACTS_SUFFIX
    ensure_dir(artifact_dir)
    rows = process_all_files(input_dir, artifact_dir)
    df = pd.DataFrame(rows)
    df.to_csv(output_csv, index=False)
    plot_histograms(df, artifact_dir)
    print(f"\nArtifacts saved in: {artifact_dir}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Infrasound Event Detector with Artifact Export")
    parser.add_argument("input_dir", help="Root directory with month/day/hour.sac files")
    parser.add_argument("output_csv", help="CSV output file")
    args = parser.parse_args()
    main(args.input_dir, args.output_csv)

