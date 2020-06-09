from sklearn.utils import shuffle
from PIL import Image


def process_ball(ball_img):
    return ball_img.resize((32, 32), Image.BILINEAR)


def process_img(img):
    w, h = img.size
    return img.resize((w//8, h//8), Image.BILINEAR)


def get_files(Dir, fileNames):
    dataset = []
    for fileName in fileNames:
        data = {}

        data['raw'] = process_img(Image.open(f"{Dir}/raw/{fileName}"))
        data['silver_ball'] = process_ball(Image.open(f"{Dir}/silver/{fileName[:-4]}-silver.jpg"))
        data['white_ball'] = process_ball(Image.open(f"{Dir}/white/{fileName[:-4]}-white.jpg"))
        data['yellow_ball'] = process_ball(Image.open(f"{Dir}/yellow/{fileName[:-4]}-yellow.jpg"))

        dataset.append(data)
    return dataset


def get_dataset():
    dataset = []
    for Dir in DATASET_DIR:
        for dirPath, dirNames, fileNames in os.walk(f"{Dir}/raw"):
            dataset.extend(get_files(Dir, fileNames))
    return dataset


if __name__ == "__main__":
    DATASET_DIR = ["indoor", "outdoor"]
    dataset = get_dataset()
