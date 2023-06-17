import cv2

img = cv2.imread("data/walmart1.png", 0)
img = cv2.resize(img, (0, 0), fx=0.5, fy=0.5)
cv2.imshow("image", img)
# cv2.waitKey(0)

ret, thresh_basic = cv2.threshold(img, 70, 255, cv2.THRESH_BINARY)
cv2.imshow("thresh", thresh_basic)
# cv2.waitKey(0)

thresh_adapt = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 119, 1)
cv2.imshow("adaptive_threshold", thresh_adapt)

if cv2.waitKey(0) & 0xff == ord("q"):
    cv2.destroyAllWindows()
