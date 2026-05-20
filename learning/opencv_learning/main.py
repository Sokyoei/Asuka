import cv2


def main():
    img = cv2.imread(r"D:\Andromeda\FSNI_logo.jpg")
    resized_img = cv2.resize(img, (128, 128))
    cv2.imshow('Lena', resized_img)
    cv2.imwrite(r"D:\Andromeda\FSNI_logo_resized.png", resized_img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
