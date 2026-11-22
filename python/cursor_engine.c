
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
    // Clamp X
    if (x < 0.0f)
        x = 0.0f;
    if (x > 1.0f)
        x = 1.0f;

    // Clamp Y
    if (y < 0.0f)
        y = 0.0f;
    if (y > 1.0f)
        y = 1.0f;

    *screen_x = (int)(x * (screen_width - 1));
    *screen_y = (int)(y * (screen_height - 1));
}
