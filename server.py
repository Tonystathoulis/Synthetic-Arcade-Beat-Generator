from flask import Flask, request, send_file
"""
Algorithmic Beat Generator Server
---------------------------------
This Flask server provides endpoints to generate algorithmic beats and serve the client UI.
Key processes:
    - / : Serves the client HTML interface.
    - /generate-beat : Generates a new beat using procedural audio or samples, then returns a WAV file.
    - Uses numpy, scipy, and pydub for audio processing.
    - Loads samples from the 'samples' directory or synthesizes them if missing.
    - Procedural generation includes drums, bass, keys, and leads.
    - See comments throughout for details on each function.
"""
import numpy as np
from scipy.io.wavfile import write as wav_write
from pydub import AudioSegment, effects
from io import BytesIO
import random, math, time, uuid, os, hashlib, glob

app = Flask(__name__)

# ==== CONFIG ====
SAMPLE_RATE = 32000
MIN_SEC = 120
MAX_SEC = 150
NORMALIZE_PEAK = 0.95
SAMPLES_DIR = "samples"  # relative to server.py

# ==== HELPERS ====
def new_entropy_seed():
    # Generates a new random seed using UUID, time, and OS entropy for reproducibility.
    entropy = f"{uuid.uuid4()}_{time.time()}_{os.urandom(16)}".encode()
    digest = hashlib.sha256(entropy).hexdigest()
    return int(digest[:8], 16)

def seconds_to_samples(sec):
    # Converts seconds to sample count based on SAMPLE_RATE.
    return int(sec * SAMPLE_RATE)

def safe_add(mix, pos, sound):
    # Safely adds a sound array to the mix at the given position, handling array bounds.
    if pos >= len(mix): return
    end = min(len(mix), pos + len(sound))
    mix[pos:end] += sound[:end - pos]

def format_filename(style):
    # Formats a unique filename for the generated beat.
    uid = uuid.uuid4().hex[:8]
    return f"beat_{style}_{uid}.wav"

# ==== ADSR ====
def _adsr_env(length, a=0.01, d=0.05, s=0.7, r=0.2):
    # Generates an ADSR envelope for shaping audio amplitude.
    n = seconds_to_samples(length)
    a_n = max(1, int(a * SAMPLE_RATE))
    d_n = max(1, int(d * SAMPLE_RATE))
    r_n = max(1, int(r * SAMPLE_RATE))
    s_n = max(0, n - (a_n + d_n + r_n))
    attack = np.linspace(0, 1, a_n, endpoint=False)
    decay = np.linspace(1, s, d_n, endpoint=False)
    sustain = np.ones(s_n) * s
    release = np.linspace(s, 0, r_n, endpoint=False)
    env = np.concatenate([attack, decay, sustain, release])
    if len(env) < n: env = np.pad(env, (0, n - len(env)))
    return env[:n]

# ==== NOTE UTILS ====
NOTES = {'C':0,'C#':1,'D':2,'D#':3,'E':4,'F':5,'F#':6,'G':7,'G#':8,'A':9,'A#':10,'B':11}
def note_to_freq(note, octave=4):
    # Converts a note name and octave to frequency in Hz (A440 standard).
    n = NOTES[note]
    midi = (octave + 1) * 12 + n
    return 440.0 * 2 ** ((midi - 69) / 12.0)

def degree_to_note(tonic, mode, degree):
    # Maps a scale degree to a note name based on tonic and mode.
    major = [0,2,4,5,7,9,11]
    minor = [0,2,3,5,7,8,10]
    scale = minor if mode == 'minor' else major
    tonic_idx = NOTES[tonic]
    semitone = (tonic_idx + scale[(degree-1)%7]) % 12
    for k,v in NOTES.items():
        if v == semitone: return k
    return 'C'

def random_progression():
    # Randomly selects a key, mode, and chord progression.
    keys = [('C','minor'),('D','minor'),('A','minor'),('F','major'),('G','minor'),('E','minor')]
    key, mode = random.choice(keys)
    progressions = [[1,6,4,5],[1,5,6,4],[1,4,1,5],[6,4,5,1],[2,5,1,4]]
    return key, mode, random.choice(progressions)

# ==== INSTRUMENT GENERATORS (procedural as before) ====
def _gen_kick_sample(decay=9.0):
    length = 0.55
    t = np.linspace(0, length, seconds_to_samples(length), endpoint=False)
    freq = 90 * np.exp(-5 * t) + 30
    wave = np.sin(2 * np.pi * freq * t + random.random()*2*np.pi)
    env = np.exp(-t * decay)
    return (wave * env).astype(np.float32)

def _gen_snare_sample(tone_amp=0.4, decay=12.0):
    length = 0.28
    t = np.linspace(0, length, seconds_to_samples(length), endpoint=False)
    noise = np.random.default_rng().normal(0,1,len(t))
    tone = tone_amp * np.sin(2*np.pi*180*t) * np.exp(-t*6)
    env = np.exp(-t*decay)
    return (noise*env*0.9 + tone).astype(np.float32)

def _gen_hat_sample(decay=80.0):
    length = 0.06
    t = np.linspace(0, length, seconds_to_samples(length), endpoint=False)
    noise = np.random.default_rng().normal(0,1,len(t))
    hp = np.concatenate(([0], np.diff(noise)))
    env = np.exp(-t*decay)
    return (hp*env*0.6).astype(np.float32)

def _gen_808_sample(freq=45.0, style='trap'):
    length = 1.6
    t = np.linspace(0,length,seconds_to_samples(length),endpoint=False)
    if style=='trap':
        base = np.sign(np.sin(2*np.pi*freq*t))*(np.abs(np.sin(2*np.pi*freq*t))**0.85)
    else:
        base = np.sin(2*np.pi*freq*t) # arcade simpler
    env = np.exp(-t*1.2)
    y = np.copy(base*env)
    for i in range(1,len(y)):
        y[i] = 0.995*y[i-1] + 0.005*y[i]
    return y.astype(np.float32)

def _gen_key_sample(freq=220.0, style='trap'):
    length = 1.2
    t = np.linspace(0,length,seconds_to_samples(length),endpoint=False)
    saw = np.zeros_like(t)
    max_harmonics = 40 if style=='arcade' else 20
    for k in range(1,max_harmonics):
        saw += (1.0/k)*np.sin(2*np.pi*freq*k*t + random.random()*2*np.pi)
    saw *= 0.35
    env = _adsr_env(length, a=0.01,d=0.12,s=0.6,r=0.6)
    return (saw*env).astype(np.float32)

def _gen_lead_sample(freq=440.0, style='trap'):
    length = 0.8
    t = np.linspace(0,length,seconds_to_samples(length),endpoint=False)
    wave = np.zeros_like(t)
    max_harmonics = 11 if style=='arcade' else random.randint(6,10)
    for k in range(1,max_harmonics,2):
        wave += (1.0/k)*np.sin(2*np.pi*freq*k*t + random.random()*2*np.pi)
    wave *= 0.3
    env = _adsr_env(length, a=0.01,d=0.05,s=0.6,r=0.2)
    return (wave*env).astype(np.float32)

# ==== RESAMPLING ====
def resample_to_freq(sample, base_freq, target_freq):
    # sample: numpy 1D float32
    if base_freq is None or base_freq <= 0:
        return sample
    ratio = target_freq / base_freq
    old_n = len(sample)
    new_n = max(1, int(old_n/ratio))
    xp = np.linspace(0,1,old_n)
    x = np.linspace(0,1,new_n)
    return np.interp(x,xp,sample).astype(np.float32)

# ==== SAMPLE LOADING HELPERS ====
def audiosegment_to_mono_np(seg: AudioSegment):
    # ensure sample rate and mono
    seg = seg.set_frame_rate(SAMPLE_RATE).set_channels(1)
    samples = np.array(seg.get_array_of_samples()).astype(np.float32)
    # normalize to -1..1 depending on sample width
    max_val = float(2 ** (8 * seg.sample_width - 1))
    samples = samples / max_val
    return samples.astype(np.float32)

def find_all_wavs(folder):
    # walk recursively and return full paths to wav files
    res = []
    for root, dirs, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(('.wav', '.flac', '.mp3', '.ogg', '.aiff')):
                res.append(os.path.join(root, f))
    return res

def load_random_wav_array(folder):
    """
    Returns numpy float32 mono array normalized to [-1,1] at SAMPLE_RATE.
    Raises FileNotFoundError if none found.
    """
    if not os.path.isdir(folder):
        raise FileNotFoundError(folder)
    wavs = find_all_wavs(folder)
    if not wavs:
        raise FileNotFoundError(folder)
    choice = random.choice(wavs)
    seg = AudioSegment.from_file(choice)
    arr = audiosegment_to_mono_np(seg)
    # ensure not silent: tiny normalization
    peak = np.max(np.abs(arr)) if arr.size else 1.0
    if peak > 0:
        arr = arr / peak
    return arr

def load_or_synth(folder_name, synth_func, default_base_freq=None):
    """
    Try to load a random sample from samples/<folder_name>/ (recursively).
    If no sample found, fall back to synth_func() which should return a numpy array float32.
    Returns tuple (array, base_freq) where base_freq indicates pitch reference for resampling.
    """
    folder = os.path.join(SAMPLES_DIR, folder_name)
    try:
        arr = load_random_wav_array(folder)
        # choose sensible base freq per instrument if not provided:
        base = default_base_freq
        return arr, base
    except FileNotFoundError:
        # fall back to synth
        if folder_name == 'kick':
            return synth_func(), None
        elif folder_name == 'snare':
            return synth_func(), None
        elif folder_name == 'hats' or folder_name == 'hat' or folder_name == 'hihat':
            return synth_func(), None
        elif folder_name == 'bass' or folder_name == '808':
            # our synth returns at base 45Hz; keep that
            return synth_func(), 45.0
        elif folder_name == 'keys' or folder_name == 'key':
            return synth_func(), 220.0
        elif folder_name == 'leads' or folder_name == 'lead':
            return synth_func(), 440.0
        else:
            return synth_func(), None

# ==== EVENT TIMELINE ====
def generate_event_timeline(length_sec, style):
    events = []
    bpm = random.randint(130,160) if style=='arcade' else random.randint(120,150)
    spb = 60.0 / bpm
    step_samps = int(spb/4*SAMPLE_RATE)
    steps = int(math.ceil(length_sec/(spb/4)))

    # Drum pattern
    for i in range(steps):
        pos = i*step_samps
        if i%4==0 and random.random()>0.05: events.append((pos,'kick',{}))
        if i%8==4 and random.random()>0.2: events.append((pos,'snare',{}))
        if random.random()>0.4: events.append((pos,'hat',{}))

    # Harmonic progression
    bars = int(math.ceil(length_sec/(spb*4)))
    tonic, mode, prog = random_progression()
    chord_roots = []
    for b in range(bars):
        deg = prog[b % len(prog)]
        note = degree_to_note(tonic, mode, deg)
        freq = note_to_freq(note, octave=3)
        chord_roots.append(freq)
        pos = int(b*spb*4*SAMPLE_RATE)
        events.append((pos,'chord',{'root':freq}))
        events.append((pos,'808',{'freq':freq}))

    # Melody/lead
    for b in range(bars*2):
        if random.random()>0.5:
            pos = int(b*spb*2*SAMPLE_RATE)
            freq = random.choice([440,493,523,587,659,698,784,880])
            dur = random.choice([0.25,0.5,0.75])
            events.append((pos,'lead',{'freq':freq,'dur':dur}))

    events.sort(key=lambda x:x[0])
    return events, bpm, chord_roots

# ==== RENDER ====
def render_from_timeline(length_sec, style):
    total_samples = seconds_to_samples(length_sec)
    mix = np.zeros(total_samples, dtype=np.float32)
    events, bpm, chord_roots = generate_event_timeline(length_sec, style)

    # Try loading sample-based instruments; fallback to synth if missing.
    # Note: folder names are expected: kick, snare, hats, bass, keys, leads
    # You said your folders are: bass, clap, hats, keys, kick, leads, snare, tom
    # We'll check those names.
    # KICK
    try:
        KICK_arr, KICK_base = load_or_synth('kick', _gen_kick_sample, None)
    except Exception:
        KICK_arr, KICK_base = _gen_kick_sample(), None

    # SNARE
    try:
        SNARE_arr, SNARE_base = load_or_synth('snare', _gen_snare_sample, None)
    except Exception:
        SNARE_arr, SNARE_base = _gen_snare_sample(), None

    # HAT
    # some users name folder 'hats' — try both
    hat_folder = 'hats' if os.path.isdir(os.path.join(SAMPLES_DIR, 'hats')) else 'hat'
    try:
        HAT_arr, HAT_base = load_or_synth(hat_folder, _gen_hat_sample, None)
    except Exception:
        HAT_arr, HAT_base = _gen_hat_sample(), None

    # BASS / 808
    try:
        BASS_arr, BASS_base = load_or_synth('bass', _gen_808_sample, 45.0)
    except Exception:
        BASS_arr, BASS_base = _gen_808_sample(), 45.0

    # KEYS
    try:
        KEY_arr, KEY_base = load_or_synth('keys', _gen_key_sample, 220.0)
    except Exception:
        KEY_arr, KEY_base = _gen_key_sample(), 220.0

    # LEAD
    try:
        LEAD_arr, LEAD_base = load_or_synth('leads', _gen_lead_sample, 440.0)
    except Exception:
        LEAD_arr, LEAD_base = _gen_lead_sample(), 440.0

    # iterate events and place sounds
    for pos, etype, params in events:
        if etype == 'kick':
            # KICK_arr may be shorter/longer; scale volume randomly
            safe_add(mix, pos, KICK_arr * (0.8 + random.random() * 0.4))
        elif etype == 'snare':
            safe_add(mix, pos, SNARE_arr * (0.6 + random.random() * 0.4))
        elif etype == 'hat':
            safe_add(mix, pos, HAT_arr * (0.3 + random.random() * 0.4))
        elif etype == '808':
            f = params['freq']
            # resample BASS_arr from BASS_base to target f
            if BASS_base:
                sub = resample_to_freq(BASS_arr, BASS_base, f)
            else:
                # synth fallback already generated at appropriate harmonic-ish base; resample from default 45
                sub = resample_to_freq(BASS_arr, 45.0, f)
            safe_add(mix, pos, sub * 0.7)
        elif etype == 'chord':
            f = params['root']
            if KEY_base:
                chord = resample_to_freq(KEY_arr, KEY_base, f) * 0.6
            else:
                chord = resample_to_freq(KEY_arr, 220.0, f) * 0.6
            safe_add(mix, pos, chord)
        elif etype == 'lead':
            f = params['freq']
            dur = params['dur']
            if LEAD_base:
                seg = resample_to_freq(LEAD_arr, LEAD_base, f)
            else:
                seg = resample_to_freq(LEAD_arr, 440.0, f)
            seg = seg[:seconds_to_samples(dur)]
            safe_add(mix, pos, seg * 0.6)

    # normalize + stereo
    mix /= max(1e-5, np.max(np.abs(mix))) * NORMALIZE_PEAK
    stereo = np.stack([
        mix * 0.98 + np.roll(mix, 200) * 0.02,
        mix * 0.98 + np.roll(mix, -200) * 0.02
    ], axis=1)
    return (stereo * 32767).astype(np.int16)

# ==== ROUTES ====
@app.route('/')
def index():
    return send_file('client.html')

@app.route('/generate-beat')
def generate_beat():
    try:
        seed = new_entropy_seed()
        random.seed(seed)
        np.random.seed(seed)

        style = request.args.get('style', 'trap')
        length_sec = random.randint(MIN_SEC, MAX_SEC)
        pcm = render_from_timeline(length_sec, style)

        bio = BytesIO()
        wav_write(bio, SAMPLE_RATE, pcm)
        bio.seek(0)
        audio = AudioSegment.from_file(bio, format="wav")
        audio = effects.normalize(audio)
        out_buf = BytesIO()
        audio.export(out_buf, format="wav")
        out_buf.seek(0)
        filename = format_filename(style)
        return send_file(out_buf, mimetype="audio/wav", as_attachment=True, download_name=filename)
    except Exception as e:
        print("Error generating beat:", e)
        return "Error generating beat", 500

if __name__ == '__main__':
    app.run(debug=True)
