
#include <stdio.h>
#include <math.h>

/*
    Convert normalized hand coordinates
    into screen pixel coordinates.
*/
void map_to_screen(
    float x,
    float y,
    int screen_width,
    int screen_height,
    int *screen_x,
    int *screen_y
)
{
    // Fixed central interaction region (20% to 80%)
    const float MIN_X = 0.20f;
    const float MAX_X = 0.80f;
    const float MIN_Y = 0.20f;
    const float MAX_Y = 0.80f;

    // Clamp coordinates to the interaction region
    if (x < MIN_X) x = MIN_X;
    if (x > MAX_X) x = MAX_X;

    if (y < MIN_Y) y = MIN_Y;
    if (y > MAX_Y) y = MAX_Y;

    // Normalize coordinates within the smaller region
    x = (x - MIN_X) / (MAX_X - MIN_X);
    y = (y - MIN_Y) / (MAX_Y - MIN_Y);

    // Map normalized coordinates to screen
    *screen_x = (int)(x * (screen_width - 1));
    *screen_y = (int)(y * (screen_height - 1));
}

/*
    Dead-zone filtering and Exponential Moving Average (EMA)
*/

// Dead-zone thresholds
#define DEAD_ZONE_X 0.004f
#define DEAD_ZONE_Y 0.004f

// EMA state
static float smooth_x = -1.0f;
static float smooth_y = -1.0f;

// Last accepted raw coordinates
static float accepted_x = -1.0f;
static float accepted_y = -1.0f;

void smooth_coordinates(
    float x,
    float y,
    float alpha,
    float *output_x,
    float *output_y
)
{
    // Initialize on the first input
    if (smooth_x < 0.0f || smooth_y < 0.0f)
    {
        smooth_x = x;
        smooth_y = y;

        accepted_x = x;
        accepted_y = y;
    }
    else
    {
        // Accept X movement only when it exceeds the dead zone
        if (fabsf(x - accepted_x) > DEAD_ZONE_X)
        {
            accepted_x = x;
        }

        // Accept Y movement only when it exceeds the dead zone
        if (fabsf(y - accepted_y) > DEAD_ZONE_Y)
        {
            accepted_y = y;
        }

        // Apply EMA smoothing
        smooth_x = alpha * accepted_x
                 + (1.0f - alpha) * smooth_x;

        smooth_y = alpha * accepted_y
                 + (1.0f - alpha) * smooth_y;
    }

    *output_x = smooth_x;
    *output_y = smooth_y;
}

/*
    Reset smoothing state when hand tracking is lost.
*/
void reset_smoothing(void)
{
    smooth_x = -1.0f;
    smooth_y = -1.0f;

    accepted_x = -1.0f;
    accepted_y = -1.0f;
}
