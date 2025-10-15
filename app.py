import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, lfilter
from pathlib import Path
from PIL import Image

# --- CSS Styling ---
st.markdown("""
<style>
    .stApp {
        background-color: #121212;
        color: #E0E0E0;
    }
    h1 {
        color: #2196F3 !important;
        font-size: 40px !important;
        text-align: center;
        font-weight: bold;
    }
    .stSelectbox label, .stNumberInput label {
        color: #E0E0E0 !important;
        font-weight: bold;
    }   
    div.stButton > button:first-child {
        background-color: #1565C0;
        color: white;
        border-radius: 8px;
        height: 3em;
        width: 100%;
        font-size: 16px;
        border: none;
    }
    div.stButton > button:first-child:hover {
        background-color: #0D47A1;
        color: white;
    }
    .stSuccess {
        background-color: #1B5E20;
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-weight: bold;
    }
    .stError {
        background-color: #B71C1C;
        color: white;
        border-radius: 8px;
        padding: 10px;
        font-weight: bold;
    }
    .stMarkdown p, .css-16huue1, .css-10trblm, .css-1offfwp {
        color: #E0E0E0 !important;
    }
    /* File uploader label */
    .stFileUploader label {
        color: #E0E0E0 !important;
        font-weight: bold;
    }

    /* Radio buttons text */
    .stRadio label {
        color: #E0E0E0 !important;
        font-weight: bold;
    }

    /* Radio options */
    .stRadio div[role="radiogroup"] > label, 
    .stRadio div[role="radiogroup"] > div > label,
    .stRadio div[role="radiogroup"] p {
        color: #E0E0E0 !important;
    }

    /* Uploader background */
    .stFileUploader {
        background-color: #1e1e1e;
        border-radius: 8px;
        padding: 8px;
    }
    /* Force white bold text for checkbox labels in dark mode */
    div[data-testid="stCheckbox"] label p {
        color: #E0E0E0 !important;
        font-weight: bold !important;
    }
    /* Make the label text of every st.multiselect white */
    div[data-testid="stMultiSelect"] > label > div > p {
        color: #E0E0E0 !important;
        font-weight: bold;  /* optional */
    }
</style>
""", unsafe_allow_html=True)

# --- ISO 17842-1:2023 Thresholds ---
THRESHOLDS = {
    'Z': [(6.0, 1), (4.0, 4.0), (3.0, 11.8), (2, 40), (1.5, float('inf'))],
    'Y': [(3.0, 1), (2.0, float('inf'))],
    'X': [(6.0, 1), (4.0, 4.0), (3.0, 11.8), (2.5, 13.5), (2.0, float('inf'))]  # diagram modified 0.5g inf
}

THRESHOLDS_neg1 = {  # base case
    'Z': [(-1.5, 3.5), (-1.1, float('inf'))],  # I.2.5
    'Y': [(-3.0, 1), (-2.0, float('inf'))],  # I.2.4
    'X': [(-1.5, float('inf'))]  # I.2.3
}

THRESHOLDS_neg2 = {  # over the shoulder restraint
    'Z': [(-1.5, 3.5), (-1.1, float('inf'))],
    'Y': [(-3.0, 1), (-2.0, float('inf'))],
    'X': [(-2.0, float('inf'))]
}


# --- loading the image ---
def safe_image_show(rel_path: str, caption: str = "", width=None, use_column_width=True):
    """Robust image display that works on Streamlit Cloud + Linux paths."""
    try:
        img_path = Path(__file__).parent / rel_path
        if not img_path.exists():
            st.info(
                f"ℹ️ Axis guide image not found at: `{rel_path}`. "
                "Please add it to your repo (case-sensitive) or provide a URL."
            )
            # Optional: uncomment this to debug what files exist:
            # st.write('CWD:', Path.cwd())
            # st.write('Here:', Path(__file__).parent)
            # st.write('Assets contents:', list((Path(__file__).parent / 'assets').glob('*')))
            return

        img = Image.open(img_path)
        # IMPORTANT: use_column_width instead of use_container_width for Streamlit Cloud
        st.image(img, caption=caption, width=width, use_column_width=use_column_width)

    except Exception as e:
        st.warning(f"Couldn't load image `{rel_path}`. Error: {e}")


# --- filter the Data ---
def butter_lowpass_filter(data, cutoff=5, fs=50, order=4):
    """
    Apply a 4-pole, single-pass Butterworth low-pass filter.

    Parameters:
    data   : array-like, raw signal
    cutoff : float, corner frequency in Hz
    fs     : float, sampling frequency in Hz
    order  : int, filter order (4-pole → order=4)

    Returns:
    filtered signal (single-pass)
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    y = lfilter(b, a, data)  # single-pass filtering
    return y


# --- Manual Safety Check ---
def check_safety(axis: str, g_force: float, duration: float, ride_type: str) -> str:
    axis = axis.upper()
    if axis not in THRESHOLDS:
        return "Invalid axis."

    if g_force >= 0:
        for limit, max_time in THRESHOLDS[axis]:
            if g_force <= limit and duration <= max_time:
                return "✅ Safe"
    elif ride_type == "Roller Coaster":
        for limit, max_time in THRESHOLDS_neg2[axis]:
            if g_force >= limit and duration <= max_time:
                return "✅ Safe"
    elif ride_type == "Sled":
        for limit, max_time in THRESHOLDS_neg1[axis]:
            if g_force >= limit and duration <= max_time:
                return "✅ Safe"
    return "⚠️ Unsafe – exceeds safety limits"


# --- Dataset Reader for IMU Format ---
def read_imu_file(uploaded_file):
    # Read raw file
    df = pd.read_csv(uploaded_file, sep=",", header=None)

    # Split time string + timestamp
    time_and_stamp = df[0].str.split(" ", expand=True)
    df.insert(0, "time_str", time_and_stamp[0])  # keep only clock time
    df = df.drop(columns=[0] + list(df.columns[5:]))  # drop old combined + unused

    # Rename columns
    df.columns = ["time_str", "acc_x", "acc_y", "acc_z"]

    # Convert time to datetime
    first_time = pd.to_datetime(df["time_str"].iloc[0], format="%H:%M:%S.%f")
    last_time = pd.to_datetime(df["time_str"].iloc[-1], format="%H:%M:%S.%f")
    total_duration = (last_time - first_time).total_seconds()

    # Time pitch and continuous timeline
    n = len(df)
    time_pitch = total_duration / (n - 1)
    df["time_sec"] = np.arange(n) * time_pitch

    # Drop original time_str
    df = df.drop(columns=["time_str"])

    return df


# --- get the admissible limit for one axis at one time ---
def admissible_limit(axis: str, g: float, ride_type: str) -> float:
    """
    Return the admissible (limit) value for the given axis at value g,
    using the correct polarity tables (positive vs negative).
    If g exceeds all admissible limits, return +inf.
    """
    pos = THRESHOLDS[axis]
    neg = THRESHOLDS_neg2[axis] if ride_type == "Roller Coaster" else THRESHOLDS_neg1[axis]

    if g >= 0:
        # smallest positive limit >= g
        limits = sorted([L for L, _ in pos])
        for L in limits:
            if g <= L:
                return L
        return float('inf')
    else:
        # “closest to zero” negative limit <= g
        limits = sorted([L for L, _ in neg])  # e.g. [-3.0, -2.0]
        for L in reversed(limits):            # e.g. -2.0 then -3.0
            if g >= L:
                return L
        return float('inf')


# --- Combined Safety Check Function ---
def check_combined_safety_all(time_s, gx, gy, gz, ride_type: str):
    """
    Combined safety check point-by-point using the three formulas (I.1–I.3):
      (gx/adm_x)^2 + (gy/adm_y)^2 <= 1
      (gx/adm_x)^2 + (gz/adm_z)^2 <= 1
      (gz/adm_z)^2 + (gy/adm_y)^2 <= 1

    Returns:
        safe_points   : list[(t, gx, gy, gz)]
        unsafe_points : list[(t, gx, gy, gz)]
    """
    safe_pts, unsafe_pts = [], []

    for i in range(len(time_s)):
        t   = float(time_s.iloc[i])
        axv = float(gx.iloc[i])
        ayv = float(gy.iloc[i])
        azv = float(gz.iloc[i])

        ax_adm = admissible_limit('X', axv, ride_type)
        ay_adm = admissible_limit('Y', ayv, ride_type)
        az_adm = admissible_limit('Z', azv, ride_type)

        # If any admissible is infinite, the point is beyond limits => unsafe
        if np.isinf(ax_adm) or np.isinf(ay_adm) or np.isinf(az_adm):
            unsafe_pts.append((t, axv, ayv, azv))
            continue

        f1 = (axv/ax_adm)**2 + (ayv/ay_adm)**2 <= 1.0
        f2 = (axv/ax_adm)**2 + (azv/az_adm)**2 <= 1.0
        f3 = (azv/az_adm)**2 + (ayv/ay_adm)**2 <= 1.0

        if f1 and f2 and f3:
            safe_pts.append((t, axv, ayv, azv))
        else:
            unsafe_pts.append((t, axv, ayv, azv))

    return safe_pts, unsafe_pts


# --- Dataset Safety Check ---
def check_series_safety(axis: str, g_series, t_series, ride_type: str, spike_series=None):
    """
    Segment by g-threshold bins (not duration), keep segments intact even if spikes occur,
    classify segments by duration vs ISO bands (pos vs neg handled separately),
    and record single-sample spikes. If spike_series is provided, spikes are detected on it
    (e.g., raw signal) while segments use g_series (e.g., filtered).
    Returns: safe_segments (list[dict]), unsafe_segments (list[dict]), spikes (list[(t, g)])
    """
    if axis not in THRESHOLDS:
        return [], [], []

    thresholds_pos = THRESHOLDS[axis]
    thresholds_neg = THRESHOLDS_neg2[axis] if ride_type == "Roller Coaster" else THRESHOLDS_neg1[axis]

    # Bin edges include 0 so bins never straddle the origin
    bounds = sorted({0.0, *[L for L,_ in (thresholds_pos + thresholds_neg)]})

    def get_bin(g):
        for i in range(len(bounds)-1):
            if bounds[i] <= g < bounds[i+1]:
                return (bounds[i], bounds[i+1])
        return (bounds[-1], float('inf')) if g >= 0 else (-float('inf'), bounds[0])

    # For spike test, use raw if provided; otherwise use the same as segment series
    spike_g = spike_series if spike_series is not None else g_series

    # Extreme allowed values for instant "magnitude" spike test
    max_pos = max((L for L,_ in thresholds_pos), default=float('inf'))
    min_neg = min((L for L,_ in thresholds_neg), default=-float('inf'))

    eps = 1e-9

    segments = []
    spikes = []
    seg_start_idx = 0
    curr_bin = get_bin(g_series.iloc[0])

    for i in range(1, len(g_series)):
        g_seg = g_series.iloc[i]
        g_spk = spike_g.iloc[i]
        b = get_bin(g_seg)

        # Detect instantaneous spike but DO NOT cut the segment
        if g_spk >= 0:
            if g_spk > max_pos + eps:
                spikes.append((t_series.iloc[i], g_spk))
        else:
            if g_spk < min_neg - eps:
                spikes.append((t_series.iloc[i], g_spk))

        # If bin changed (crossed a limit boundary), close previous segment
        if b != curr_bin:
            g_slice = g_series.iloc[seg_start_idx:i]
            t_slice = t_series.iloc[seg_start_idx:i]
            segments.append({
                'start':   t_slice.iloc[0],
                'end':     t_slice.iloc[-1],
                'duration': t_slice.iloc[-1] - t_slice.iloc[0],
                'g_min':   g_slice.min(),
                'g_max':   g_slice.max(),
                'bin':     curr_bin
            })
            seg_start_idx = i
            curr_bin = b

    # Final segment
    g_slice = g_series.iloc[seg_start_idx:]
    t_slice = t_series.iloc[seg_start_idx:]
    segments.append({
        'start':   t_slice.iloc[0],
        'end':     t_slice.iloc[-1],
        'duration': t_slice.iloc[-1] - t_slice.iloc[0],
        'g_min':   g_slice.min(),
        'g_max':   g_slice.max(),
        'bin':     curr_bin
    })

    # Classify segments using correct rule per polarity
    safe_segments, unsafe_segments = [], []
    for seg in segments:
        gmin, gmax, dur = seg['g_min'], seg['g_max'], seg['duration']
        low, high = seg['bin']
        is_positive_bin = (low >= 0 and high >= 0)

        if is_positive_bin:
            # POS: need g_max ≤ limit and duration ≤ max_time
            ok = any((gmax <= L + eps) and (dur <= Tmax + eps) for L, Tmax in thresholds_pos)
        else:
            # NEG: need g_min ≥ limit and duration ≤ max_time
            ok = any((gmin >= L - eps) and (dur <= Tmax + eps) for L, Tmax in thresholds_neg)

        (safe_segments if ok else unsafe_segments).append(seg)


    return safe_segments, unsafe_segments, spikes


# --- Title ---
st.title("Vibration & Acceleration Safety Checker ⚙️")

# --- Mode Selection ---
mode = st.radio("Select Mode", ["Manual Input", "Upload Dataset"])

# --- Manual Mode ---
if mode == "Manual Input":
    st.subheader("Axis Guide Overview")

    safe_image_show("assets/Axis_Guide.png", caption="3-axis X-Y-Z")

    st.write("""
    This diagram shows how the X, Y, and Z axes are oriented 
    relative to the human body while seated in the amusement vehicle.  
    It is used as a guide when interpreting acceleration data 
    from the mounted sensors.  
    """)
    ride_type = st.selectbox("Ride Vehicle", ["Roller Coaster", "Sled"])
    axis = st.selectbox("Axis", ["X", "Y", "Z"])
    g_force = st.number_input("G-force (g)", value=1.0)
    duration = st.number_input("Duration (seconds)", value=1.0)

    if st.button("Check Safety"):
        result = check_safety(axis, g_force, duration, ride_type)
        if "Safe" in result:
            st.success(f"Result: {result}")
        else:
            st.error(f"Result: {result}")

# --- Dataset Mode ---
else:
    ride_type = st.selectbox("Ride Vehicle", ["Roller Coaster", "Sled"])
    uploaded_file = st.file_uploader("Upload IMU CSV/TXT file", type=["csv", "txt"])

    if uploaded_file is not None:
        df = read_imu_file(uploaded_file)
        
        # UI: choose which axes to invert
        st.subheader("Sensor Orientation")
        invert_axes = st.multiselect(
            "Invert (multiply by -1) the following axes",
            ['X', 'Y', 'Z'],
            default=['X', 'Z']  # make this [] if we don't want a default
        )

        # Apply inversion on raw data BEFORE filtering
        for ax in invert_axes:
            df[f'acc_{ax.lower()}'] = -df[f'acc_{ax.lower()}']
            
        fs = 50  # your IMU sampling rate
        df['acc_x_filtered'] = butter_lowpass_filter(df['acc_x'], cutoff=5, fs=fs, order=4)
        df['acc_y_filtered'] = butter_lowpass_filter(df['acc_y'], cutoff=5, fs=fs, order=4)
        df['acc_z_filtered'] = butter_lowpass_filter(df['acc_z'], cutoff=5, fs=fs, order=4)
        
        df_preview = df[["time_sec", "acc_x_filtered", "acc_y_filtered", "acc_z_filtered"]]
        st.write("### Data Preview", df_preview.head(20))
        # check if you wanna see the whole data
        if st.checkbox("Show All Data After Processing", value=False):
            st.write("### Data Preview", df_preview)
            
        results = {}
        total_unsafe_duration = 0.0

        for axis in ['X', 'Y', 'Z']:
            safe_segments, unsafe_segments, spikes = check_series_safety(
                axis,
                df[f'acc_{axis.lower()}_filtered'],   # filtered → segmentation
                df['time_sec'],
                ride_type,
                spike_series=df[f'acc_{axis.lower()}']  # raw → spike detection
            )
            results[axis] = {
                "safe": safe_segments,
                "unsafe": unsafe_segments,
                "spikes": spikes
            }
            # Calculate total unsafe duration based on the traditional safety check
            total_unsafe_duration += sum(seg['duration'] for seg in unsafe_segments)  # sum durations
            
        # Combined check (do it once for all three series)
        comb_safe, comb_unsafe = check_combined_safety_all(
            df['time_sec'],
            df['acc_x_filtered'],
            df['acc_y_filtered'],
            df['acc_z_filtered'],
            ride_type
        )
        results["combined"] = {"safe": comb_safe, "unsafe": comb_unsafe}

        # Results per axis
        st.subheader("Results")
        for axis in ['X', 'Y', 'Z']:
            safe_segments = results[axis]["safe"]
            unsafe_segments = results[axis]["unsafe"]
            comb_unsafe = results["combined"]["unsafe"]   # Combined result

            # Display safe/unsafe for traditional segments
            if len(unsafe_segments) == 0:
                st.success(f"{axis}-axis: ✅ Safe (all {len(safe_segments)} segments within thresholds)")
            else:
                total_axis_unsafe = sum(seg['duration'] for seg in unsafe_segments)  # ← fix
                st.error(
                    f"{axis}-axis: ⚠️ {len(unsafe_segments)} unsafe segments "
                    f"(total {total_axis_unsafe:.2f} s beyond limits)"
                )
            
            # Display combined unsafe check
            if axis == 'Z':
                if len(comb_unsafe) == 0:
                    st.success("Combined check (I.1&I.3): ✅ Safe (no unsafe points)")
                else:
                    st.error(f"Combined check (I.1&I.3): ⚠️ {len(comb_unsafe)} unsafe points")


        # Overall
        if all(len(results[a]["unsafe"]) == 0 for a in ['X', 'Y', 'Z']) and len(results["combined"]["unsafe"]) == 0:
            st.success("Overall: ✅ Safe")
        else:
            st.error(
                f"Overall: ⚠️ Unsafe – per-axis unsafe duration: {total_unsafe_duration:.2f} s; "
                f"combined unsafe points: {len(results['combined']['unsafe'])}"
            )

        # Plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["time_sec"], df['acc_x_filtered'], label='X-axis g', color='cyan')
        ax.plot(df["time_sec"], df['acc_y_filtered'], label='Y-axis g', color='orange')
        ax.plot(df["time_sec"], df['acc_z_filtered'], label='Z-axis g', color='green')

        # Highlight unsafe segments
        for axis_name, color in zip(['X', 'Y', 'Z'], ['cyan', 'orange', 'green']):
            unsafe_segments = results[axis_name]["unsafe"]
            for seg in unsafe_segments:
                ax.axvspan(seg['start'], seg['end'], color='red', alpha=0.3)

        # Plot spike points as red dots
        for axis_name, signal_color in zip(['X', 'Y', 'Z'], ['cyan', 'orange', 'green']):
            spikes = results[axis_name]["spikes"]
            g_vals = df[f'acc_{axis_name.lower()}_filtered']
            for t_spike, g_spike in spikes:
                ax.plot(t_spike, g_spike, 'ro', markersize=5)

        ax.axhline(0, color='white', linestyle='--', linewidth=0.8)
        ax.set_title("Acceleration (g) over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("g-force")
        ax.legend()
        st.pyplot(fig)
        
        # Table of unsafe segments
        rows = [] 
        for axis in ['X', 'Y', 'Z']:
            for seg in results[axis]["unsafe"]:
                rows.append({
                    "Axis": axis,
                    "Start (s)": seg['start'],
                    "End (s)": seg['end'],
                    "Duration (s)": seg['duration'],
                    "g_min": seg['g_min'],
                    "g_max": seg['g_max']
                })


        if rows:
            st.write("### Unsafe Segments")
            st.dataframe(pd.DataFrame(rows))
        # Table of spikes
        spike_rows = []
        for axis in ['X', 'Y', 'Z']:
            for t_spike, g_spike in results[axis]["spikes"]:
                spike_rows.append({
                    "Axis": axis,
                    "Time (s)": t_spike,
                    "G-force": g_spike,
                    "Direction": "Positive" if g_spike > 0 else "Negative"
                })

        if spike_rows:
            st.write("### Detected Spikes (Above Threshold Magnitudes)")
            st.dataframe(pd.DataFrame(spike_rows))
        # Table of all segments
        if st.checkbox("Show All Segments Table (Safe + Unsafe)", value=False):
            all_segments = []
            for axis in ['X', 'Y', 'Z']:
                for seg in results[axis]["safe"]:
                    all_segments.append({
                        "Axis": axis,
                        "Start (s)": seg['start'],
                        "End (s)": seg['end'],
                        "Duration (s)": seg['duration'],
                        "g_min": seg['g_min'],
                        "g_max": seg['g_max'],
                        "Status": "Safe"
                    })
                for seg in results[axis]["unsafe"]:
                    all_segments.append({
                        "Axis": axis,
                        "Start (s)": seg['start'],
                        "End (s)": seg['end'],
                        "Duration (s)": seg['duration'],
                        "g_min": seg['g_min'],
                        "g_max": seg['g_max'],
                        "Status": "Unsafe"
                    })

            if all_segments:
                st.write("### All Segments (Safe and Unsafe)")
                st.dataframe(pd.DataFrame(all_segments))
                
                
        # Combined safety plot
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(df["time_sec"], df['acc_x_filtered'], label='X-axis g', color='cyan')
        ax.plot(df["time_sec"], df['acc_y_filtered'], label='Y-axis g', color='orange')
        ax.plot(df["time_sec"], df['acc_z_filtered'], label='Z-axis g', color='green')

        # Highlight unsafe segments based on combined check
        for (t_comb, gxv, gyv, gzv) in results["combined"]["unsafe"]:
            ax.axvline(x=t_comb, color='red', linewidth=1.0, alpha=0.7)  # red thin vertical line for each unsafe point

        ax.set_title("Combined Safety Check (X, Y, Z) over Time")
        ax.set_xlabel("Time (s)")
        ax.set_ylabel("g-force")
        ax.legend()
        st.pyplot(fig)
        
        # Also add combined safety status in the output tables
        combined_rows = [
            {"Time (s)": t, "G-force X": gx, "G-force Y": gy, "G-force Z": gz, "Status": "Unsafe"}
            for (t, gx, gy, gz) in results["combined"]["unsafe"]
        ]
        if combined_rows:
            st.write("### Detected Combined Unsafe Accelerations")

            st.dataframe(pd.DataFrame(combined_rows))


