import pytesseract


class OCR:

    def __init__(self, imagefile_or_array=None, tesseract_cmd_path=r'/usr/bin/tesseract'):
        self.tesseract_cmd_path = tesseract_cmd_path
        self.image = imagefile_or_array

    def set_tesseract_cmd_path(self, path=None):
        if path is not None:
            pytesseract.pytesseract.tesseract_cmd = path
            return
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd_path

    def extract_text_to_string(self, imagefile=None):
        return pytesseract.image_to_string(self.image)

    def bulk_extraction(self, filepath_containing_filenames):
        return pytesseract.image_to_string(filepath_containing_filenames)

    # Get bounding box estimates
    def get_bounding_box(self, imagefile=None):
        pytesseract.image_to_boxes(self.image)

    # Get verbose data including boxes, confidences, line and page numbers
    def get_image_data(self, imagefile=None):
        pytesseract.image_to_data(self.image)

    def write_to_csv(self, filepath):
        text_lines = [line for line in self.extract_text_to_string().split('\n') if line and not line.isspace()]
        with open(file=filepath , mode="a") as f:
            for line in text_lines:
                f.write(f"{line}\n")
