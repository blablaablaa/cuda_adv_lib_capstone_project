# Using CUDA for edge detection

**Capstone Project for the Coursera
[GPU Programming Specialization](https://www.coursera.org/specializations/gpu-programming)**


My **learning objectives** for this project:

- learn about cuDNN
- learn about CUDA graphs
- understand performance impact of various steps

## Table of Contents

- [Key Concepts](#key-concepts)
- [Project Overview and Build Instructions](#project-overview-and-build-instructions)


## Key Concepts

The application performs multiple steps:

1. Loading the data from disk (either image, or video)
2. Manipulate image or single frames from the video
3. Store manipulated data back to disk, either as an image, or as a video file.

The image processing makes use of several CUDA kernels and cuDNN convolutions:

1. Convert `uint8` quantized image to `float`
   [`convertUint8ToFloat`](./include/image_manip.hpp) (One should consider
   implementing the filter completely in `int` quantization for potential performance
   improvements.)
2. Convert RGB image to grayscale using a cuDNN convolution operator
   ([`m_conv_to_grayscale`](./include/filter.hpp#L30))
3. Apply a sobel operator to detect edges: cuDNN convolution to find horizontal
   and vertical edges [`m_conv_edges`](./include/filter.hpp#L32),
   [`pointwiseAbs`](./include/image_manip.hpp) ensures all values are positive,
   and another cuDNN conv to reduce the 2-channel image with vertical and horizontal
   edges to a 1-channel image representing all edges
   [`m_conv_reduce_2d_to_1d`](./include/filter.hpp#L39)
4. Looping n times, applying another cuDNN convolution to smooth the edges
   [`m_conv_smooth`](./include/filter.hpp#L40)
   while limiting the intensity using [`pointwiseMin`](./include/image_manip.hpp)
5. Applying a convolution [`m_conv_delete`](./include/filter.hpp#L46): the
   convolution kernel is designed to reduce intensity in areas without intensity
   variations. It is essentially a variant of an edge detection kernel. Afterward,
   another [`pointwiseMin`](./include/image_manip.hpp) introduces an intensity
   cutoff.
6. A convolution [`m_conv_broadcast_to_4_channels`](./include/filter.hpp#L31) transforms
   the 1-channel grayscale edge image into a 4-channel RGBA image,
   [`pointwiseHalo`](./include/image_manip.hpp) combines the edge image with
   the initial color image, and [`setChannel`](./include/image_manip.hpp) sets the
   intensity of the alpha channel to the maximum value.
7. Finally, convert the floating point image back to `uint8`
   [`convertFloatToUint8`](./include/image_manip.hpp).

The filter's code can be found in `Filter::runFilterOnGpu` in
[`filter.hpp`](./include/filter.hpp#L80). To simplify the relatively verbous
implementation of a cuDNN convolution, I introduced the class
[`Convolution`](./include/convolution.hpp). The CUDA kernels underlying the
various functions like [`convertUint8ToFloat`](./include/image_manip.hpp),
[`convertFloatToUint8`](./include/image_manip.hpp),
[`m_conv_to_grayscale`](./include/filter.hpp#L30),
[`pointwiseAbs`](./include/image_manip.hpp),
[`pointwiseMin`](./include/image_manip.hpp),
[`pointwiseHalo`](./include/image_manip.hpp), and
[`setChannel`](./include/image_manip.hpp), are implemented in [`cuda_kernels.cu`](./src/cuda_kernels.cu)

The project is structured into several additional files. The class
[`Cli`](./include/cli.hpp) manages the command line interface. The class
[`CudaGraph`](./include/cuda_graph.hpp) abstracts the capturing and the execution
of a CUDA graph. [`GpuBlob`](./include/gpu_blob.hpp) handles GPU memory
management, including allocation and deallocation, as well as transfer between CPU
and GPU at a low level. It is used by the [`ImageGPU`](./include/types.hpp#L51)
class and the [`Kernel`](./include/types.hpp#L11) class.
[`GpuSession`](./include/gpu_session.hpp) manages session handles. Several helper
functions have been from the NVidia
[cuda samples](https://github.com/NVIDIA/cuda-samples).
[`io.hpp`](./include/io.hpp) handles file i/o, and, last but not least,
[`Timer`](./include/timer.hpp) provides a simple class for performing runtime
measurements.

The main program is implemented in [`edgeDetection.cpp`](./src/edgeDetection.cpp).
The discussed CUDA graph offers no advantage over imperative execution
when processing only a single image. Therefore, I implemented a graph only in
[`processVideo`](./src/edgeDetection.cpp#L64), which reads from a video
clip, applies the graph frame-by-frame, and writes the frames back into a new video
clip.

## Project Overview and Build Instructions

### Code Organization

`bin/`
After building using make, this folder will hold the executable `edgeDetection` together with a number of object files.

`data/`
This folder holds example data, an image [`Lena.png`](./data/Lena.png). By default, the output is stored in the same folder.

`include/`
Holds header files.

`src/`
Holds the source code of the project

`.clang-format`, `.clang-tidy`
Config files for setting up clang tools that can be used for formatting and
cleaning the code.

`README.md`
This file describes the project. It holds human-readable instructions for building and executing the code.

`Makefile`
This file contains the instructions for building the project using the make utility. It specifies the dependencies, compilation flags, and the target executable to be generated.

### Supported OSes

The project was testet on Ubuntu 24.04.

### Supported CPU Architecture

The project was tested on x86_64.

### CUDA APIs involved

NVIDIA CUDA Deep Neural Network library (cuDNN)

### Dependencies needed to build/run

- [FreeImage](https://freeimage.sourceforge.io/) On Ubuntu, `apt install libfreeimage-dev`
- [cuDNN](https://docs.nvidia.com/deeplearning/cudnn/latest/) Need to [install NVidia cuDNN Backend](https://docs.nvidia.com/deeplearning/cudnn/installation/latest/linux.html)
- [opencv](https://opencv.org/) `apt install libopencv-dev`

For using tools like clang-format and clang-tidy, you need to install separate
packages

### Prerequisites

Download and install the [CUDA Toolkit](https://developer.nvidia.com/cuda-downloads) for your corresponding platform. The project was tested with CUDA 12.5.
Make sure the dependencies mentioned in [Dependencies]() section above are installed.

### Tools

- `bear` for creating a `compile_commands.json` file. This can be used for clangd integration in vscode.
- `clang-tidy`, `clang-format` as linter and for code formatting

### Build and Run

The project has been tested on Ubuntu 24.04. There is a [`Makefile`](./Makefile), therefore the project can be built using

```
$ make all
```

### Running the Program

You can run the program using the following command:

```bash
make run
```

This command will execute the compiled binary, applying the edge filter on the example input image (Lena.png), and save the result as Lena_edge.png in the data/ directory.

If you wish to run the binary directly with custom input/output files, you can use:

```bash
./bin/edgeDetection --input data/Lena.png --output data/Lena_edges.png
```

You can also process a video file. The i/o is implemented using opencv:

```bash
./bin/edgeDetection --input some_input.mp4 --output edges_video.mp4
```

Cleaning Up: To clean up the compiled binaries and other generated files, run:

```bash
make clean
```

This will remove all files in the bin/ directory.


##The Video result 
https://drive.google.com/file/d/1MNbIKuu8FbBDxNHocPlZPX_vlck4GJKw/view?usp=sharing
