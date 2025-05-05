# How to compile and run
- Compile the program with:
```bash
nvcc -o color_gradient color_gradient.cu
```

- Run the program with:
```bash
./color_gradient [width] [height]
```

- For example:
```bash
./color_gradient 1920 1080
```
This will create a 1920x1080 image. If no arguments are provided, it defaults to 800x600.

# Program explanation
1. The program generates an RGB image with a smooth color gradient from left to right.

2. The color scheme transitions from red on the left side to blue on the right side, with green varying in the middle to create a smooth rainbow-like effect.

3. Each pixel's color is computed in parallel using CUDA threads, with each thread responsible for calculating one pixel's RGB values.

4. The image is saved as a PPM (Portable PixMap) file, which is a simple uncompressed image format that can be opened by most image viewers or converted to other formats.

5. The program measures and reports the execution time using CUDA events.

You can view the resulting image using various image viewers or convert it to other formats using tools like ImageMagick:

