import numpy as np
from grabcut import GCClient
import cv2

path = 'F:/code/tasks/Grab Cut/img/llama.jpg'
img = cv2.imread(path, cv2.IMREAD_COLOR)
component_count = 5

output = np.zeros(img.shape, np.uint8)

GC = GCClient(img, component_count)
cv2.namedWindow('output')
cv2.namedWindow('input')
a = cv2.setMouseCallback('input', GC.init_mask)
cv2.moveWindow('input', img.shape[0] + 100, img.shape[1] + 100)

count = 0
print("Instructions: \n")
print("Draw a rectangle around the object using right mouse button \n")
print('Press N to continue \n')

while True:
    cv2.imshow('output', output)
    cv2.imshow('input', np.asarray(GC.img, dtype=np.uint8))
    k = 0xFF & cv2.waitKey(1)

    if k == 27:
        break

    elif k == ord('n'):
        if GC.rect_or_mask == 0:
            GC.run()
            GC.rect_or_mask = 1
        elif GC.rect_or_mask == 1:
            GC.iter(1)

    FGD = np.where((GC._mask == 1) + (GC._mask == 3), 255, 0).astype('uint8')

    output = cv2.bitwise_and(GC.img2, GC.img2, mask=FGD)

cv2.destroyAllWindows()
