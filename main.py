from image_line_reader import OCR
from preprocess import PreProcess

preprocess = PreProcess()
img = preprocess.read_frame(filepath="data/walmart1.png")
gray = preprocess.get_grayscale(img)
# denoised = preprocess.denoise(gray)
# binary_image = preprocess.adaptiveThreshold(denoised)
# thresh = preprocess.threshold(gray)
ocr = OCR(imagefile_or_array=gray)
ocr.set_tesseract_cmd_path()

line = ocr.extract_lines()
print(line)
print(len(line))
print(len(ocr.get_lines_container))
print(ocr.get_lines_container)
print(len(ocr.get_bounding_box_container))
print(ocr.get_bounding_box_container)

ocr.draw_bounding_box_for_lines(line, image=img)
preprocess.show_frame(img)
ocr.write_to_csv(filepath="out.csv")
