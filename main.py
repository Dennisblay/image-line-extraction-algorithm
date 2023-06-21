from image_line_reader import LineReader

from preprocess import PreProcess

if __name__ == '__main__':
    preprocess = PreProcess()
    img = preprocess.read_frame(filepath="data/img.jpeg")
    # img = preprocess.resize_by_half(img)

    ocr = LineReader(imagefile_or_array=img)
    line = ocr.extract_bounding_box_for_lines()
    ocr.draw_bounding_box_for_lines(line, image=ocr.image)
    preprocess.show_frame(ocr.image)
    ocr.write_to_csv(filepath="out.csv")
