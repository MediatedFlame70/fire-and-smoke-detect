"""Image geometry shared by training, validation, and inference."""

from PIL import Image


def letterbox_image(image: Image.Image, bbox=None, size: int = 448, fill=(124, 116, 104)):
    """Resize without distorting aspect ratio and pad to a square.

    ``bbox``, when supplied, is normalized [cx, cy, width, height] relative to
    the original image. The returned box is relative to the padded square.
    """
    width, height = image.size
    scale = min(size / width, size / height)
    resized_width = max(1, round(width * scale))
    resized_height = max(1, round(height * scale))
    resampling = getattr(Image, "Resampling", Image).BILINEAR
    resized = image.resize((resized_width, resized_height), resampling)
    left = (size - resized_width) // 2
    top = (size - resized_height) // 2

    canvas = Image.new("RGB", (size, size), fill)
    canvas.paste(resized, (left, top))
    if bbox is None:
        return canvas

    cx, cy, box_width, box_height = (float(value) for value in bbox)
    adjusted_bbox = [
        (cx * resized_width + left) / size,
        (cy * resized_height + top) / size,
        box_width * resized_width / size,
        box_height * resized_height / size,
    ]
    return canvas, adjusted_bbox

