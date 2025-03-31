# Handwritten Character Image Merging Program

## Introduction
This program uses Python and the PIL (Pillow) library to randomly select character images from a handwritten character dataset and merge multiple handwritten character images into a single image. It is suitable for applications that require generating random handwritten character images, such as font display or handwriting recognition research.

## Features
- Remove white borders from images
- Horizontally merge multiple images with customizable spacing
- Support line breaks
- Support for random selection of character images 
- Resize images to a specified height while maintaining aspect ratio (options)

## Prerequisites
1. Python 3.x
2. Pillow library

### Install Pillow
```
pip install pillow
```

## Dataset Format
- Traditional-Chinese-Handwriting-Dataset-master folder: Contains traditional Chinese handwriting data https://github.com/AI-FREE-Team/Traditional-Chinese-Handwriting-Dataset
- MNIST_data folder: Contains handwritten digit data
- Image file naming format: `<character>_<number>.png`

## Usage
### Run the Program Demo
```bash
python main.py
```

### Output Image
After running the program, the output image will be saved as `output.png`.

### Customize Input String
Modify the following code in `main.py` to change the output string:
```python
image = test_generate_handwriten_image(
    "本中文手寫字集是由南臺科技大學電子系所提供",
    row_spacing_range=(5, 10),
    col_spacing_range=(1, 3),
    max_columns=10,
    resize=True,
)
```

### Generate Randomly Arranged Image

You can use the function generate_randomly_arranged_image(images, canvas_size) to create a new image with randomly placed images from the input list without overlap. The function returns both the merged image and the list of image positions.
If the return of function has None empty, then this is location fail.

## Configuration
- `row_spacing_range`: Random spacing range between images in the same row
- `col_spacing_range`: Random spacing range between images in different rows
- `max_columns`: Maximum number of characters per row
- `resize`: Whether to resize images based on height

## Notes
- The folder structure of the image dataset must be correct; otherwise, the program cannot find the character images.
- If an unrecognized character is encountered, the program will skip it.

## License
This program and dataset are provided by the Department of Electronic Engineering, Southern Taiwan University of Science and Technology, for academic research purposes only.