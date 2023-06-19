import cv2
import pytesseract


class OCR:

    def __init__(self, imagefile_or_array, tesseract_cmd_path=r'/usr/bin/tesseract'):
        self.tesseract_cmd_path = tesseract_cmd_path
        self.image = imagefile_or_array

    def set_tesseract_cmd_path(self, path=None):
        if path is not None:
            pytesseract.pytesseract.tesseract_cmd = path
            return
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd_path

    def extract_text_to_string(self):
        return pytesseract.image_to_string(self.image)

    def bulk_extraction(self, filepath_containing_filenames):
        return pytesseract.image_to_string(filepath_containing_filenames)

    # Get bounding box estimates
    def get_bounding_box_string(self):
        return pytesseract.image_to_boxes(self.image)

    def draw_bounding_box_for_each_character(self):
        image = self.image
        h, w, c = image.shape
        boxes = pytesseract.image_to_boxes(image)
        for b in boxes.splitlines():
            b = b.split(' ')
            cv2.rectangle(image, (int(b[1]), h - int(b[2])), (int(b[3]), h - int(b[4])), (0, 255, 0), 2)

    # Get verbose data including boxes, confidences, line and page numbers
    def get_image_data(self):
        return pytesseract.image_to_data(self.image)

    def write_to_csv(self, filepath):
        text_lines = [line for line in self.extract_text_to_string().split('\n') if line and not line.isspace()]
        with open(file=filepath, mode="a") as f:
            for line in text_lines:
                f.write(f"{line}\n")

    def draw_bounding_box_for_lines(self, bounding_boxes_container):
        image = self.image
        h, w, c = image.shape
        for b in bounding_boxes_container:
            cv2.rectangle(image, (int(b[0]), h - int(b[1])), (int(b[2]), h - int(b[3])), (0, 255, 0), 2)

    def extract_lines(self):
        lines_first_bound = None  # To temporarily store the bounding box of the first character on a line
        lines_last_bound = None  # To temporarily store the bounding box of the first character on a line
        count = 0  # To keep track of which character we're on

        lines_bounding_box = []  # To store bounding box of each line
        bounding_box_container = self.get_bounding_box_container
        lines_container = self.get_lines_container
        # try:
        for line_index, line in enumerate(lines_container):
            line_length = len(line)  # for line in lines
            for char_index, char in enumerate(line):  # for character in line
                if char.isspace() or not char:
                    continue
                if lines_first_bound is None:
                    lines_first_bound = bounding_box_container[count]
                if line_length == char_index + 1:
                    lines_last_bound = bounding_box_container[count]
                    lines_bounding_box.append(
                        (*lines_first_bound[:2], *lines_last_bound[2:], f"line-{line_index}"))
                count += 1

            lines_first_bound = None

        # except IndexError:
        #     pass
        print(count)
        return lines_bounding_box

    @property
    def get_characters(self):
        characters = [extract.split()[:-1][0] for extract in self.get_bounding_box_string().split("\n") if
                      (extract and not extract.isspace() and extract.split()[:-1][0] != "~")]
        return characters

    @property
    def get_bounding_box_container(self):
        bounding_boxes = [extract.split()[1:-1] for extract in self.get_bounding_box_string().split("\n") if
                          extract and not extract.isspace() and extract.split()[:-1][0] != "~"]
        return bounding_boxes

    @property
    def get_lines_container(self):
        lines = [line for line in self.extract_text_to_string().split("\n") if (line and not line.isspace())]
        return lines
