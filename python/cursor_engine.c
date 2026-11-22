
#include <stdio.h>

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

static float smooth_x = -1.0f;
static float smooth_y = -1.0f;

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
    }
    else
    {
        smooth_x = alpha * x + (1.0f - alpha) * smooth_x;
        smooth_y = alpha * y + (1.0f - alpha) * smooth_y;
    }

    *output_x = smooth_x;
    *output_y = smooth_y;
}

