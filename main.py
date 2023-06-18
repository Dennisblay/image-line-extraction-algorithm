from ocr import OCR
from preprocess import PreProcess

preprocess = PreProcess()
img = preprocess.read_frame(filepath="data/walmart1.png")
gray = preprocess.get_grayscale(img)
denoised = preprocess.denoise(gray)
binary_image = preprocess.adaptiveThreshold(denoised)

ocr = OCR(imagefile_or_array=img)
ocr.set_tesseract_cmd_path()

# line = ocr.extract_line()
# print(line)
# print(len(line))
# print(len(ocr.get_lines_container))
# print(ocr.get_lines_container)
# print(len(ocr.get_bounding_box_container))
# print(ocr.get_bounding_box_container)
# print(len(ocr.get_characters))
# print(ocr.get_characters)
# # preprocess.show_frame(img)
# ocr.write_to_csv("out.csv")
