# All the knobs live here. run_pipeline.py imports this and nothing else
# should hardcode a date/threshold/window - if you find one, move it here.

# --- what to pull ---
# dev range was one week: "2024-08-01 00:00" .. "2024-08-08 00:00"
# 80 days below is what's cached. Oct 20-30 kept dying on NOAA so I stopped
# there. bump TEND when you want more - old chunks get reused.
TSTART = "2024-08-01 00:00"
TEND   = "2024-10-20 00:00"

SATELLITE  = 16          # stick to one. mixing GOES-16/18 shifts the calibration
RESOLUTION = "flx1s"     # flx1s = 1s, avg1m = 1min

# --- chunking ---
# 5 months of 1s data is ~13M rows. can't hold that + derived cols at once,
# so do it in chunks and only keep the downsampled result.
CHUNK_DAYS   = 10
LEADIN_HOURS = 8         # extra history so the 4h background isn't cold at a chunk start

# NOAA falls over if you hammer it. default is 5 parallel conns; 2 is slower
# but stopped giving me the truncated files that then wedge sunpy's cache.
MAX_CONN    = 2
FETCH_RETRY = 4

# no timeout = infinite hang when the server accepts then goes silent. learned that one.
DL_TIMEOUT_TOTAL = 900   # whole batch
DL_TIMEOUT_READ  = 60    # silence on the socket before we bail

# --- cleaning ---
GAP_LIMIT_S = 60         # interpolate gaps up to a minute, mask anything longer
BG_WINDOW   = "4h"
BG_Q        = 0.05       # 5th pct ~ quiet floor
FLOOR       = 1e-9       # floor before log10 so we never hit log(0)

# --- features ---
CADENCE    = "10s"       # what the model sees. storage stays 1s.
WINDOW_MIN = 30
STRIDE_S   = 60
SMOOTH_MIN = 2.0         # trailing smoothing on the rate feature

# --- labels ---
LABEL_SCHEME = "efold"   # catalogued | efold | background
EFOLD_MIN    = 12.6      # from notebook 04, roughly class-independent
K_EFOLD      = 2.0       # decay label runs to peak + K*efold

# 5min horizon keeps ~98% of positives nowcastable. 10 -> 83%, 15 -> 66%.
HORIZON_MIN    = 5
MIN_WARN_CLASS = 2       # M and up

# --- split ---
TRAIN_FRAC = 0.70
VAL_FRAC   = 0.85        # val = 0.70..0.85, test = the rest
