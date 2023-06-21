import cv2
import pytesseract


class LineReader:

    def __init__(self, imagefile_or_array, tesseract_cmd_path=r'/usr/bin/tesseract'):
        self.tesseract_cmd_path = tesseract_cmd_path
        self.image = imagefile_or_array
        try:
            rotated = self.rotate_image()
            if rotated is not None:
                self.image = rotated
        except pytesseract.TesseractError:
            print("Invalid Resolution")
            self.image = imagefile_or_array

        self.lines_container = []
        self.bounding_boxes_lines_container = []
        self.character_container = []

    def set_tesseract_cmd_path(self, path=None):
        if path is not None:
            pytesseract.pytesseract.tesseract_cmd = path
            return
        pytesseract.pytesseract.tesseract_cmd = self.tesseract_cmd_path

    def rotate_image(self):
        angle_of_rotation = self.detect_text_rotation()
        if angle_of_rotation < 0:
            return None

            # Get the image dimensions
        height, width = self.image.shape[:2]

        # Calculate the rotation matrix
        rotation_matrix = cv2.getRotationMatrix2D((width / 2, height / 2), -angle_of_rotation, 1)

        # Perform the rotation
        rotated_image = cv2.warpAffine(self.image, rotation_matrix, (width, height))
        return rotated_image

    def detect_text_rotation(self):
        # Load the image

        text = pytesseract.image_to_osd(self.image)

        # Extract the rotation angle from the OCR result
        rotation_angle = 0
        print(text)
        for line in text.split('\n'):
            if 'Rotate: ' in line:
                rotation_angle = float(line.split(': ')[-1])
                break

        return rotation_angle

    def extract_text_to_string(self):

        extracted_strings = pytesseract.image_to_string(self.image)
        return extracted_strings

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

    def draw_bounding_box_for_lines(self, bounding_boxes_container, image=None):
        if image is None:
            image = self.image
        for b in bounding_boxes_container:
            b = b.split(",")
            cv2.rectangle(image, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (0, 255, 0), 2)

    # Get verbose data including boxes, confidences, line and page numbers
    def get_image_data(self):
        return pytesseract.image_to_data(self.image)

    def write_to_csv(self, filepath):

        with open(file=filepath, mode="a") as f:
            f.write("TEXT,TOP_LEFT_X,TOP_LEFT_Y,BOTTOM_RIGHT_X,BOTTOM_RIGHT_Y,lINE_NUMBER \n")
            for line, bound in zip(self.lines_container, self.bounding_boxes_lines_container):
                for _line, _bound in zip(line, bound):
                    f.write(f"{_line}, {_bound}\n")

    def extract_bounding_box_for_lines(self):  # Returns bounding box of lines
        lines_first_bound = None  # To temporarily store the bounding box of the first character on a line
        lines_last_bound = None  # To temporarily store the bounding box of the first character on a line
        count = 0  # To keep track of which character we're on
        try:
            h, w, c = self.image.shape
        except ValueError:
            h, w = self.image.shape
        lines_bounding_box = []  # To store bounding box of each line
        bounding_box_container = self.get_bounding_box_container
        lines_container = self.get_lines_container
        try:
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
                            f"{int(lines_first_bound[0])},{h - int(lines_first_bound[1])},{int(lines_last_bound[2])},{h - int(lines_last_bound[3])},line-{line_index}")
                    count += 1

                lines_first_bound = None

        except IndexError:
            pass
        self.bounding_boxes_lines_container.append(lines_bounding_box)
        return lines_bounding_box

    @property
    def get_characters_in_container(self):

        characters = [extract.split()[:-1][0] for extract in self.get_bounding_box_string().split("\n") if
                      (extract and not extract.isspace() and extract.split()[:-1][0] != "~")]
        self.character_container.append(characters)
        return characters

    @property
    def get_bounding_box_container(self):
        bounding_boxes = [extract.split()[1:-1] for extract in self.get_bounding_box_string().split("\n") if
                          extract and not extract.isspace() and extract.split()[:-1][0] != "~"]
        return bounding_boxes

    @property
    def get_lines_container(self):
        lines = [line for line in self.extract_text_to_string().split("\n") if (line and not line.isspace())]
        self.lines_container.append(lines)
        return lines
