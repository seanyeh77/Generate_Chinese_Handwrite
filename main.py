import os
from PIL import Image
import random

paths_of_char = dict()


# 去除圖片上下左右白邊
def trim_white_borders(image):
    image = image.convert("RGBA")
    width, height = image.size
    pixels = image.load()

    top = 0
    while top < height and all(
        pixels[x, top] == (255, 255, 255, 255) for x in range(width)
    ):
        top += 1

    bottom = height - 1
    while bottom > top and all(
        pixels[x, bottom] == (255, 255, 255, 255) for x in range(width)
    ):
        bottom -= 1

    left = 0
    while left < width and all(
        pixels[left, y] == (255, 255, 255, 255) for y in range(height)
    ):
        left += 1

    right = width - 1
    while right > left and all(
        pixels[right, y] == (255, 255, 255, 255) for y in range(height)
    ):
        right -= 1

    trimmed_image = image.crop((left, top, right + 1, bottom + 1))
    return trimmed_image


def resize_to_height(image, target_height):
    width, height = image.size
    new_width = int(width * (target_height / height))
    resized_image = image.resize((new_width, target_height), Image.LANCZOS)
    return resized_image


def save_image(image, output_path):
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    ext = os.path.splitext(output_path)[1].lower()
    if ext == ".jpg" or ext == ".jpeg":
        image = image.convert("RGB")  # JPEG 不支援 alpha channel
        image.save(output_path, format="JPEG")
    elif ext == ".png":
        image.save(output_path, format="PNG")
    else:
        raise ValueError(f"Unsupported file extension: {ext}")


def resize_to_height(img, target_height) -> Image.Image:
    """Resize the image to the specified height while maintaining aspect ratio."""
    aspect_ratio = img.width / img.height
    new_width = int(target_height * aspect_ratio)
    return img.resize((new_width, target_height))


def merge_images_horizontally(
    images,
    resize=False,
    row_spacing_range=(0, 0),
    col_spacing_range=(0, 0),
    max_columns: int = None,
) -> Image.Image:
    """
    Merge a list of images horizontally with random spacing between them.
    Arguments:
        images: list of PIL Image objects to be merged.
        resize: bool, whether to resize the images to the same height.
        row_spacing_range: tuple, range of random spacing between rows.
        col_spacing_range: tuple, range of random spacing between columns.
        max_columns: int, maximum number of columns in the output image.
    Returns:
        merged_image: PIL Image object of the merged image.
    """
    if max_columns is None or max_columns < 1:
        max_columns = len(images)
    rows = [images[i : i + max_columns] for i in range(0, len(images), max_columns)]
    for i, row in enumerate(rows):
        max_height = max(img.height for img in row)
        if resize:
            row = [resize_to_height(img, max_height) for img in row]
        rows[i] = row

    row_heights = []
    row_widths = []
    row_spacing_lists = [
        [random.randint(*row_spacing_range) for _ in range(len(row) - 1)] + [0]
        for row in rows
    ]
    column_spacing_lists = [
        random.randint(*col_spacing_range) for _ in range(len(rows) - 1)
    ] + [0]
    for i, row in enumerate(rows):
        max_height = max(img.height for img in row)

        total_width = sum(img.width for img in row) + sum(row_spacing_lists[i])

        row_heights.append(max_height)
        row_widths.append(total_width)

    merged_width = max(row_widths)
    merged_height = sum(row_heights) + sum(column_spacing_lists)

    merged_image = Image.new(
        "RGBA", (merged_width, merged_height), (255, 255, 255, 255)
    )

    y_offset = 0
    for i in range(len(rows)):
        x_offset = 0
        for j in range(len(rows[i])):
            merged_image.paste(
                rows[i][j],
                (x_offset, y_offset + row_heights[i] // 2 - rows[i][j].height // 2),
            )
            x_offset += rows[i][j].width + row_spacing_lists[i][j]

        y_offset += row_heights[i] + column_spacing_lists[i]

    return merged_image


def test_generate_handwriten_image(
    input: str,
    resize=False,
    row_spacing_range=(0, 0),
    col_spacing_range=(0, 0),
    max_columns: int = None,
) -> Image.Image:
    """
    Generate a handwritten image from the input string.
    Arguments:
        input: str, the input string to be converted into a handwritten image.
        resize: bool, whether to resize the images to the same height.
        row_spacing_range: tuple, range of random spacing between rows.
        col_spacing_range: tuple, range of random spacing between columns.
        max_columns: int, maximum number of columns in the output image.
    Returns:
        image: PIL Image object of the generated handwritten image.
    """

    paths_of_char: dict = get_dict_of_char()

    # check if input is a list of strings
    images = []
    for key in input:
        if key not in paths_of_char:
            print(f"Warning: '{key}' not found in dataset.")
            return None
        else:
            path = random.choice(paths_of_char[key])
            image = Image.open(path)
            image = trim_white_borders(image)
            images.append(image)

    image = merge_images_horizontally(
        images, resize, row_spacing_range, col_spacing_range, max_columns
    )
    return image


def get_dict_of_char() -> dict:
    if paths_of_char:
        return paths_of_char
    #  Traditional-Chinese-Handwriting-Dataset-master
    for folder in os.listdir(
        os.path.join(
            os.path.dirname(__file__),
            "data",
            "Traditional-Chinese-Handwriting-Dataset-master",
            "data",
        )
    ):
        if not folder.endswith(".zip"):
            for file in os.listdir(
                os.path.join(
                    os.path.dirname(__file__),
                    "data",
                    "Traditional-Chinese-Handwriting-Dataset-master",
                    "data",
                    folder,
                    "cleaned_data(50_50)",
                )
            ):
                if file.split("_")[0] not in paths_of_char:
                    paths_of_char[file.split("_")[0]] = []
                paths_of_char[file.split("_")[0]].append(
                    os.path.join(
                        os.path.dirname(__file__),
                        "data",
                        "Traditional-Chinese-Handwriting-Dataset-master",
                        "data",
                        folder,
                        "cleaned_data(50_50)",
                        file,
                    )
                )

    # MNIST_data
    for folder in os.listdir(
        os.path.join(os.path.dirname(__file__), "data", "MNIST_data")
    ):
        for file in os.listdir(
            os.path.join(os.path.dirname(__file__), "data", "MNIST_data", folder)
        ):
            if file.split("_")[0] not in paths_of_char:
                paths_of_char[file.split("_")[0]] = []
            paths_of_char[file.split("_")[0]].append(
                os.path.join(
                    os.path.dirname(__file__), "data", "MNIST_data", folder, file
                )
            )
    return paths_of_char

def is_valid_text(text: str) -> bool:
    """Check if the text is valid."""
    paths_of_char: dict = get_dict_of_char()
    for char in text:
        if char not in paths_of_char:
            return False
    return True


def generate_randomly_arranged_image(images, canvas_size):
    """ "
    "Generate a randomly arranged image with the given images."
    Arguments:
        images: list of PIL Image objects to be arranged randomly.
        canvas_size: tuple (width, height) for the canvas size.
    Returns:
        canvas: PIL Image object with the randomly arranged images.
        positions: list of tuples (x, y, x + width, y + height) for each image.
    """
    canvas = Image.new("RGBA", canvas_size, (255, 255, 255, 255))
    positions = []
    for img in images:
        if img is None:
            positions.append(None)
            continue
        max_attempts = 1000
        if img.width > canvas_size[0] or img.height > canvas_size[1]:
            positions.append(None)
            continue
        while max_attempts > 0:
            x = random.randint(0, canvas_size[0] - img.width)
            y = random.randint(0, canvas_size[1] - img.height)
            overlap = False

            for pos in positions:
                if pos is None:
                    continue
                if not (
                    x + img.width < pos[0]
                    or x > pos[0] + pos[2]
                    or y + img.height < pos[1]
                    or y > pos[1] + pos[3]
                ):
                    overlap = True
                    break

            if not overlap:
                canvas.paste(img, (x, y), img)
                positions.append((x, y, x + img.width, y + img.height))
                break
            max_attempts -= 1
            
        if max_attempts == 0:
            positions.append(None)

    return canvas, positions


if __name__ == "__main__":
    image1 = test_generate_handwriten_image(
        "這是第一行",
        row_spacing_range=(5, 10),
        col_spacing_range=(1, 3),
        max_columns=10,
        resize=False,
    )
    image2 = test_generate_handwriten_image(
        "本中文手寫字集是由南臺科技大學電子系所提供",
        row_spacing_range=(5, 10),
        col_spacing_range=(1, 3),
        max_columns=None,
        resize=False,
    )
    image3 = test_generate_handwriten_image(
        "本中文手寫",
        row_spacing_range=(5, 10),
        col_spacing_range=(1, 3),
        max_columns=10,
        resize=False,
    )
    image, postion = generate_randomly_arranged_image(
        [image1, image2, image3], (800, 200)
    )
    save_image(image, "output.png")
