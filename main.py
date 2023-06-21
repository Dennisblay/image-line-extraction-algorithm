from image_line_reader import LineReader

from preprocess import PreProcess

if __name__ == '__main__':
    preprocess = PreProcess()
    img = preprocess.read_frame(filepath="data/walmart1.png")
    img = preprocess.resize_by_half(img)
    ocr = LineReader(imagefile_or_array=img)
    ocr.set_tesseract_cmd_path()
    line = ocr.extract_bounding_box_for_lines()

    ocr.draw_bounding_box_for_lines(line, image=img)
    preprocess.show_frame(img)
    ocr.write_to_csv(filepath="out.csv")
