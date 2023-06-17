from ocr import OCR
from preprocess import PreProcess

preprocess = PreProcess()
img = preprocess.read_frame(filepath="data/walmart1.png")
gray = preprocess.get_grayscale(img)
denoised = preprocess.denoise(gray)
binary_image = preprocess.adaptiveThreshold(denoised)

# preprocess.show_frame(binary_image)
ocr = OCR(imagefile_or_array="data/walmart1.png")
ocr.set_tesseract_cmd_path()
print(ocr.get_bounding_box())
ocr.write_to_csv("out.csv")

