import albumentations as A
import cv2


def alb_transforms(images, bboxes):
    transform = A.Compose([
        A.Blur(p=0.3),
        A.HorizontalFlip(p=0.5),
        A.Rotate,
        A.RandomBrightnessContrast(p=0.2),
    ], bbox_params=A.BboxParams(format='pascal_voc'))
    numpy_imgs = images.detach().cpu().numpy()
    transformed = transform(image=images, bboxes=bboxes)
    transformed_image = transformed['image']
    transformed_bboxes = transformed['bboxes']
    return transformed_image, transformed_bboxes

