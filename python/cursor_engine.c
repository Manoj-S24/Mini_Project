/*
    cursor_engine.c

    Drop-in replacement. The three functions main.py already uses keep the
    exact same names and signatures:

        map_to_screen(x, y, screen_width, screen_height, *screen_x, *screen_y)
        smooth_coordinates(x, y, alpha, *output_x, *output_y)
        reset_smoothing()

    Build (MinGW-w64, 64-bit to match your 64-bit Python):
        gcc -O2 -shared -o cursor_engine.dll cursor_engine.c

    Build (MSVC, x64 Native Tools prompt):
        cl /LD /O2 cursor_engine.c /Fe:cursor_engine.dll

    Close the running app first; Windows locks a loaded DLL.
*/

#include <math.h>
#include <stddef.h>

#ifdef _WIN32
#include <windows.h>
#define EXPORT __declspec(dllexport)
#else
#include <time.h>
#define EXPORT
#endif

#define PI_F 3.14159265358979f


/* ------------------------------------------------------------------
   Timing
   ------------------------------------------------------------------ */

static double now_seconds(void)
{
#ifdef _WIN32
    static LARGE_INTEGER frequency;
    static int have_frequency = 0;
    LARGE_INTEGER counter;

    if (!have_frequency)
    {
        QueryPerformanceFrequency(&frequency);
        have_frequency = 1;
    }

    QueryPerformanceCounter(&counter);
    return (double)counter.QuadPart / (double)frequency.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}


/* ------------------------------------------------------------------
   Screen mapping
   ------------------------------------------------------------------ */

/* Central interaction region of the camera frame (20% to 80%).
   Adjustable at runtime with set_active_region(). */
static float region_min_x = 0.20f;
static float region_max_x = 0.80f;
static float region_min_y = 0.20f;
static float region_max_y = 0.80f;

static float clamp_float(float value, float low, float high)
{
    if (value < low)  return low;
    if (value > high) return high;
    return value;
}

/*
    Convert normalized hand coordinates into screen pixel coordinates.
*/
EXPORT void map_to_screen(
    float x,
    float y,
    int screen_width,
    int screen_height,
    int *screen_x,
    int *screen_y
)
{
    if (screen_x == NULL || screen_y == NULL)
        return;

    if (screen_width < 1)  screen_width = 1;
    if (screen_height < 1) screen_height = 1;

    /* Ignore NaN / infinity from a bad tracking frame. */
    if (!isfinite(x)) x = 0.5f;
    if (!isfinite(y)) y = 0.5f;

    /* Clamp to the interaction region. */
    x = clamp_float(x, region_min_x, region_max_x);
    y = clamp_float(y, region_min_y, region_max_y);

    /* Normalize within the smaller region. */
    float nx = (x - region_min_x) / (region_max_x - region_min_x);
    float ny = (y - region_min_y) / (region_max_y - region_min_y);

    /* Map to screen, rounding to the nearest pixel (the old code
       truncated, which biased the cursor slightly toward the top-left). */
    *screen_x = (int)(nx * (float)(screen_width  - 1) + 0.5f);
    *screen_y = (int)(ny * (float)(screen_height - 1) + 0.5f);
}

/*
    Optional: change the interaction region at runtime (values 0..1,
    max must be greater than min). Not required by main.py.
*/
EXPORT void set_active_region(float min_x, float max_x, float min_y, float max_y)
{
    if (!isfinite(min_x) || !isfinite(max_x) ||
        !isfinite(min_y) || !isfinite(max_y))
        return;

    if (min_x < 0.0f || max_x > 1.0f || max_x - min_x < 0.05f) return;
    if (min_y < 0.0f || max_y > 1.0f || max_y - min_y < 0.05f) return;

    region_min_x = min_x;
    region_max_x = max_x;
    region_min_y = min_y;
    region_max_y = max_y;
}


/* ------------------------------------------------------------------
   Smoothing: One Euro filter (replaces dead zone + fixed EMA)

   Slow hand  -> strong smoothing -> no jitter, precise clicking.
   Fast hand  -> light smoothing  -> no lag.
   It is time-aware, so it behaves the same at any FPS.

   The `alpha` argument from main.py (SMOOTHING_ALPHA) is kept and now acts
   as the maximum weight given to the newest sample, i.e. the fastest the
   cursor is ever allowed to respond. 0.8 behaves like your old setting
   during fast movement.
   ------------------------------------------------------------------ */

/* Tunable (or change at runtime with set_smoothing_params()). */
static float min_cutoff = 1.2f;   /* Hz. Lower = smoother when slow  */
static float beta       = 8.0f;   /* Higher = less lag when moving fast */
static float d_cutoff   = 1.0f;   /* Hz. Smoothing of the speed estimate */

/* Filter state. */
static int    initialized = 0;
static float  smooth_x = 0.0f;
static float  smooth_y = 0.0f;
static float  raw_prev_x = 0.0f;
static float  raw_prev_y = 0.0f;
static float  speed = 0.0f;       /* filtered speed, normalized units / s */
static double last_time = 0.0;

static float alpha_from_cutoff(float cutoff, float dt)
{
    float tau = 1.0f / (2.0f * PI_F * cutoff);
    return 1.0f / (1.0f + tau / dt);
}

EXPORT void smooth_coordinates(
    float x,
    float y,
    float alpha,
    float *output_x,
    float *output_y
)
{
    if (output_x == NULL || output_y == NULL)
        return;

    /* Bad input: keep the last good output. */
    if (!isfinite(x) || !isfinite(y))
    {
        *output_x = initialized ? smooth_x : 0.5f;
        *output_y = initialized ? smooth_y : 0.5f;
        return;
    }

    if (!isfinite(alpha)) alpha = 0.8f;
    alpha = clamp_float(alpha, 0.01f, 1.0f);

    double t = now_seconds();

    /* Initialize on the first input after a reset. */
    if (!initialized)
    {
        smooth_x = x;
        smooth_y = y;
        raw_prev_x = x;
        raw_prev_y = y;
        speed = 0.0f;
        last_time = t;
        initialized = 1;

        *output_x = smooth_x;
        *output_y = smooth_y;
        return;
    }

    float dt = (float)(t - last_time);
    last_time = t;

    /* Guard against duplicate timestamps and long pauses. */
    dt = clamp_float(dt, 1.0f / 240.0f, 0.2f);

    /* Hand speed (shared by X and Y so diagonals behave the same). */
    float raw_speed = hypotf(x - raw_prev_x, y - raw_prev_y) / dt;
    float a_speed = alpha_from_cutoff(d_cutoff, dt);
    speed = a_speed * raw_speed + (1.0f - a_speed) * speed;

    /* Faster hand -> higher cutoff -> less smoothing. */
    float cutoff = min_cutoff + beta * speed;
    float a = alpha_from_cutoff(cutoff, dt);

    if (a > alpha)
        a = alpha;

    smooth_x = a * x + (1.0f - a) * smooth_x;
    smooth_y = a * y + (1.0f - a) * smooth_y;

    raw_prev_x = x;
    raw_prev_y = y;

    *output_x = smooth_x;
    *output_y = smooth_y;
}

/*
    Optional: tune smoothing at runtime. Not required by main.py.
*/
EXPORT void set_smoothing_params(float new_min_cutoff, float new_beta, float new_d_cutoff)
{
    if (isfinite(new_min_cutoff) && new_min_cutoff > 0.0f) min_cutoff = new_min_cutoff;
    if (isfinite(new_beta)       && new_beta >= 0.0f)      beta       = new_beta;
    if (isfinite(new_d_cutoff)   && new_d_cutoff > 0.0f)   d_cutoff   = new_d_cutoff;
}

/*
    Reset smoothing state when hand tracking is lost.
*/
EXPORT void reset_smoothing(void)
{
    initialized = 0;
    smooth_x = 0.0f;
    smooth_y = 0.0f;
    raw_prev_x = 0.0f;
    raw_prev_y = 0.0f;
    speed = 0.0f;
    last_time = 0.0;
}