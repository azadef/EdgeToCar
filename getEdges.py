import os, cv2

imDir = "../bmw10_ims"
resultDir = "../bmw10_edges"

images = os.listdir(imDir)

for imPath in images:
  image = cv2.imread(os.path.join(imDir, imPath))
  gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
  blurred = cv2.GaussianBlur(gray, (3, 3), 0)
  edges = cv2.Canny(blurred, 200, 250)
  
  cv2.imwrite(os.path.join(resultDir,imPath), edges)
